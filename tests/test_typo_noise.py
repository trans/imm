"""Typo noise: substitutions drawn from a unigram distribution, the model
reading a noisy copy (input_ids) while its targets stay the clean ids, and
make_typo_text.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.typo_noise import substitute, unigram_probs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import V, corpus, make_model, ragged_segments, runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SubstituteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = np.random.default_rng(0).integers(0, 10, 200_000)
        self.probs = unigram_probs(self.ids, 12)  # two vocabulary entries never occur

    def test_rate_and_only_real_changes(self) -> None:
        noisy, changed = substitute(self.ids, 0.05, self.probs, np.random.default_rng(1))
        differs = noisy != self.ids
        self.assertEqual(int(differs.sum()), changed)  # every chosen position changed
        self.assertAlmostEqual(changed / self.ids.size, 0.05, delta=0.002)
        self.assertTrue(set(np.unique(noisy[differs])) <= set(range(10)))  # replacements follow the unigram

    def test_seeded_and_zero_rate(self) -> None:
        a, _ = substitute(self.ids, 0.02, self.probs, np.random.default_rng(7))
        b, _ = substitute(self.ids, 0.02, self.probs, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        same, changed = substitute(self.ids, 0.0, self.probs, np.random.default_rng(7))
        np.testing.assert_array_equal(same, self.ids)
        self.assertEqual(changed, 0)

    def test_refuses_a_one_character_distribution(self) -> None:
        with self.assertRaises(ValueError):
            substitute(self.ids, 0.1, np.eye(12)[3], np.random.default_rng(0))


class InputIdsTest(unittest.TestCase):
    """segment_loss and segment_loss_streams read input_ids and score ids."""

    STREAMS = 3

    def setUp(self) -> None:
        self.ids, self.rows = corpus()
        self.segments = ragged_segments(len(self.ids) - 1)
        self.model = make_model()
        self.model.eval()
        # The last position each path scores and never reads: the end of its
        # last segment (split_streams drops segments that don't fill a round).
        self.last_target = {
            self.loss_single: self.segments[-1][1],
            self.loss_streams: runner.split_streams(self.segments, self.STREAMS)[-1][-1][1],
        }

    def loss_single(self, ids, input_ids=None):
        with torch.no_grad():
            offset = self.segments[0][0]
            loss, *_ = self.model.segment_loss(
                ids, self.segments, [], 4, carry_hidden=True, prior_offset=offset, input_ids=input_ids, **self.rows
            )
        return float(loss)

    def loss_streams(self, ids, input_ids=None):
        stream_segments = runner.split_streams(self.segments, self.STREAMS)
        spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
        with torch.no_grad():
            loss, *_ = self.model.segment_loss_streams(
                ids, stream_segments, None, 4, carry_hidden=True, input_ids=input_ids,
                **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()},
            )
        return float(loss)

    def changed_at(self, position):
        other = self.ids.clone()
        other[position] = (other[position] + 1) % V
        return other

    def test_input_ids_equal_to_ids_change_nothing(self) -> None:
        for loss in (self.loss_single, self.loss_streams):
            self.assertEqual(loss(self.ids, input_ids=self.ids.clone()), loss(self.ids))

    def test_the_model_reads_input_ids(self) -> None:
        noisy = self.changed_at(10)
        for loss in (self.loss_single, self.loss_streams):
            self.assertNotAlmostEqual(loss(self.ids, input_ids=noisy), loss(self.ids), places=5)

    def test_targets_come_from_ids(self) -> None:
        # Differs only at the last position scored, which is never read:
        # reading it changes nothing, scoring it does.
        for loss in (self.loss_single, self.loss_streams):
            last = self.changed_at(self.last_target[loss])
            self.assertAlmostEqual(loss(self.ids, input_ids=last), loss(self.ids), places=5)
            self.assertNotAlmostEqual(loss(last), loss(self.ids), places=5)

    def test_streams_match_single_with_noisy_input(self) -> None:
        noisy = self.ids.clone()
        noisy[::7] = (noisy[::7] + 3) % V
        model = make_model()
        model.train()
        for streams in (1, 3):
            stream_segments = runner.split_streams(self.segments, streams)
            want = 0.0
            for segs in stream_segments:
                offset, end = segs[0][0], segs[-1][1]
                loss, *_ = model.segment_loss(
                    self.ids, segs, [], 4, carry_hidden=True, prior_offset=offset, input_ids=noisy,
                    **{k: v[offset:end] for k, v in self.rows.items()},
                )
                want += float(loss)
            spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
            got, *_ = model.segment_loss_streams(
                self.ids, stream_segments, None, 4, carry_hidden=True, input_ids=noisy,
                **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()},
            )
            self.assertAlmostEqual(float(got), want, places=3)


class MakeTypoTextTest(unittest.TestCase):
    def test_writes_an_aligned_copy_and_its_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            text = "the cat sat on the mat.\n" * 400
            (tmp / "in.txt").write_text(text)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "make_typo_text.py"), "--input", str(tmp / "in.txt"),
                 "--reference", str(tmp / "in.txt"), "--vocab-input", str(tmp / "in.txt"), "--rate", "0.05",
                 "--seed", "3", "--output", str(tmp / "out.txt")],
                check=True, capture_output=True,
            )
            out = (tmp / "out.txt").read_text()
            meta = json.loads((tmp / "out.json").read_text())
            self.assertEqual(len(out), len(text))
            self.assertEqual(sum(a != b for a, b in zip(out, text)), meta["chars_changed"])
            self.assertAlmostEqual(meta["chars_changed"] / len(text), 0.05, delta=0.01)
            self.assertTrue(set(out) <= set(text))


if __name__ == "__main__":
    unittest.main()
