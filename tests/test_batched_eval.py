"""--eval-streams: the held-out segments scored as batched streams, each
warmed up on unscored segments first. With a burn-in covering the whole text
it is the one-stream score; with less, every character is still scored once."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import V, corpus, make_model, ragged_segments, runner  # noqa: E402


class WindowsTest(unittest.TestCase):
    def test_scored_runs_partition_and_windows_share_a_length(self) -> None:
        for count, streams, burn_in in [(10, 3, 0), (10, 3, 2), (10, 3, 100), (7, 7, 1), (5, 9, 2), (100, 8, 5)]:
            windows = runner.eval_stream_windows(count, streams, burn_in)
            self.assertEqual(len(windows), min(streams, count))
            scored = [i for _, _, lo, hi in windows for i in range(lo, hi)]
            self.assertEqual(scored, list(range(count)))
            widths = {hi - lo for lo, hi, _, _ in windows}
            self.assertEqual(len(widths), 1)
            for lo, hi, s_lo, s_hi in windows:
                self.assertTrue(0 <= lo <= s_lo < s_hi <= hi <= count)
                self.assertGreaterEqual(s_lo - lo, min(burn_in, s_lo))  # warmed up on burn_in, or from the start
            sizes = [s_hi - s_lo for _, _, s_lo, s_hi in windows]
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_refusals(self) -> None:
        for args in [(0, 2, 0), (5, 0, 0), (5, 2, -1)]:
            with self.assertRaises(ValueError):
                runner.eval_stream_windows(*args)


class EvaluateStreamsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=300)
        self.segments = ragged_segments(len(self.ids) - 1)
        self.model = make_model()

    def score(self, **kw):
        return runner.evaluate(
            self.model, self.ids, self.segments, 4, carry_hidden=True, feedback_state="gru", token_feedback="none",
            **self.rows, **kw,
        )

    def test_full_burn_in_is_the_one_stream_score(self) -> None:
        nll, ppl, bpc, stats = self.score()
        for streams, chunk in [(3, 64), (5, 7), (4, 1)]:
            got_nll, _, _, got = self.score(streams=streams, burn_in=len(self.segments), chunk=chunk)
            self.assertAlmostEqual(got_nll, nll, places=5)
            self.assertEqual(got["token_count"], stats["token_count"])
            for key in ("first_token_loss", "later_token_loss", "residual_alpha_sum", "context_norm_sum"):
                self.assertAlmostEqual(got[key], stats[key], places=3, msg=key)
            for key in ("first_token_count", "later_token_count", "residual_alpha_count"):
                self.assertEqual(got[key], stats[key], msg=key)
            self.assertAlmostEqual(got["gate_sum"] / got["gate_count"], stats["gate_sum"] / stats["gate_count"], places=6)

    def test_short_burn_in_scores_every_character_once(self) -> None:
        nll, *_ , stats = self.score()
        for burn_in in (0, 4):
            got_nll, *_, got = self.score(streams=6, burn_in=burn_in)
            self.assertEqual(got["token_count"], stats["token_count"])
            self.assertLess(abs(got_nll - nll), 0.2)  # cold starts cost a little, not a different text

    def test_chunking_changes_nothing(self) -> None:
        a = self.score(streams=4, burn_in=3, chunk=64)[0]
        b = self.score(streams=4, burn_in=3, chunk=2)[0]
        self.assertAlmostEqual(a, b, places=5)

    def test_reads_input_ids(self) -> None:
        noisy = self.ids.clone()
        noisy[::5] = (noisy[::5] + 1) % V
        want = self.score(input_ids=noisy)[0]
        got = self.score(input_ids=noisy, streams=3, burn_in=len(self.segments))[0]
        self.assertAlmostEqual(got, want, places=5)
        self.assertNotAlmostEqual(got, self.score(streams=3, burn_in=len(self.segments))[0], places=4)

    def test_eval_mode_and_training_path_unchanged(self) -> None:
        self.model.train()
        self.score(streams=3, burn_in=2)
        self.assertFalse(self.model.training)  # evaluate leaves it in eval mode, as before
        # The training call (no mask, no stats) is the old code path: losses match a masked call scoring all.
        stream_segments = runner.split_streams(self.segments, 3)
        spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
        rows = {k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()}
        with torch.no_grad():
            plain, n, *_ = self.model.segment_loss_streams(self.ids, stream_segments, None, 4, carry_hidden=True, **rows)
            mask = np.ones((3, len(stream_segments[0])), dtype=bool)
            masked, m, *_ = self.model.segment_loss_streams(
                self.ids, stream_segments, None, 4, carry_hidden=True, score_mask=mask, **rows
            )
        self.assertEqual(n, m)
        self.assertAlmostEqual(float(plain), float(masked), places=3)


if __name__ == "__main__":
    unittest.main()
