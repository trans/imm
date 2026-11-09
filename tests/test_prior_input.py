"""--prior-input: the residual sees what the prior predicts (log-probs), what each
context length's counts say (depth-evidence), or both."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

V = 7
D = 3  # evidence depths; each row is D x (V + 1)


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_memory_model.py"
    spec = importlib.util.spec_from_file_location("run_segment_memory_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner_module()


def make_model(prior_input: str, seed: int = 0, prior_strength: float = 1.0, width: int | None = None):
    if width is None:
        width = D * (V + 1) if prior_input in ("depth-evidence", "both") else 0
    torch.manual_seed(seed)
    model = runner.SegmentMemoryLM(
        V, 8, 8,
        use_rope=True, rope_positions="char", memory_state="terminal", memory_record="written",
        rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn", attention_layers=1,
        attention_heads=2, input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0,
        prior_strength=prior_strength, prior_residual_scale=1.0, prior_residual_l2=0.0, prior_residual_gate="none",
        prior_residual_gate_max=2.0, prior_residual_gate_features=0, prior_residual_state_features=[],
        skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
        record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0,
        utility_aux_weight=0.0, utility_temperature=1.0, prior_input=prior_input,
        prior_evidence_width=width,
    )
    model.zero_prediction_heads()
    return model


def data(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (25,), generator=g)
    rows = torch.log_softmax(torch.randn(24, V, generator=g), dim=-1)  # row r predicts ids[r + 1]
    return ids, [(0, 5), (5, 9), (9, 16), (16, 24)], rows


def evidence_rows(seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(24, D, V + 1, generator=g)


def loss(model, ids, segments, rows, offset=0, evidence=None):
    total, _, _, _, _ = model.segment_loss(
        ids, segments, [], 4, carry_hidden=True, prior_log_probs=rows, prior_offset=offset, prior_evidence=evidence
    )
    return total


class PriorInputTest(unittest.TestCase):
    def test_other_parameters_start_the_same(self) -> None:
        plain, seeing = make_model("none"), make_model("log-probs")
        theirs = dict(seeing.named_parameters())
        for name, p in plain.named_parameters():
            torch.testing.assert_close(theirs[name], p, rtol=0, atol=0, msg=name)
        # and leaves the random stream where it was, for whatever draws next
        make_model("none")
        after_plain = torch.rand(3)
        make_model("log-probs")
        torch.testing.assert_close(torch.rand(3), after_plain, rtol=0, atol=0)
        self.assertEqual(set(theirs) - set(dict(plain.named_parameters())),
                         {"prior_input_proj.0.weight", "prior_input_proj.0.bias", "prior_input_proj.1.weight", "prior_input_proj.1.bias"})

    def test_first_pass_is_unchanged(self) -> None:
        # The prediction heads start at zero, so the loss alone can't tell;
        # the GRU's inputs must match too.
        ids, segments, rows = data()
        inputs = {}
        for kind in ("none", "log-probs"):
            model = make_model(kind)
            inputs[kind] = []
            model.rnn.register_forward_pre_hook(lambda _m, args, kind=kind: inputs[kind].append(args[0]))
            with torch.no_grad():
                inputs[kind].append(loss(model, ids, segments, rows))
        for a, b in zip(inputs["none"], inputs["log-probs"], strict=True):
            torch.testing.assert_close(b, a, rtol=0, atol=0)

    def test_training_moves_the_projection_and_then_the_rows_matter(self) -> None:
        # At prior strength 0 the rows reach the loss only through the input.
        ids, segments, rows = data()
        other = rows.flip(-1)
        with torch.no_grad():
            plain = make_model("none", prior_strength=0.0)
            self.assertEqual(loss(plain, ids, segments, rows).item(), loss(plain, ids, segments, other).item())
        model = make_model("log-probs", prior_strength=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            loss(model, ids, segments, rows).backward()
            opt.step()
        self.assertGreater(model.prior_input_proj[1].weight.abs().sum().item(), 0.0)
        with torch.no_grad():
            self.assertNotAlmostEqual(loss(model, ids, segments, rows).item(), loss(model, ids, segments, other).item(), places=4)

    def test_position_r_gets_row_r(self) -> None:
        # Row r is the prior's prediction of ids[r + 1], from context up to
        # ids[r]: the GRU input at position r is embed(ids[r]) + proj(row r).
        ids, segments, rows = data()
        model = make_model("log-probs")
        with torch.no_grad():
            model.prior_input_proj[1].weight.normal_()
        seen = []
        model.rnn.register_forward_pre_hook(lambda _m, args: seen.append(args[0].squeeze(0)))
        with torch.no_grad():
            loss(model, ids, segments[1:], rows[5:], offset=5)
            want = model.embed(ids[5:24]) + model.prior_input_proj(rows[5:24])
        torch.testing.assert_close(torch.cat(seen), want)

    def test_rows_line_up_with_an_offset(self) -> None:
        ids, _, rows = data()
        model = make_model("log-probs")
        with torch.no_grad():
            model.prior_input_proj[1].weight.normal_()
        later = [(9, 16), (16, 24)]
        torch.testing.assert_close(loss(model, ids, later, rows[9:], offset=9), loss(model, ids, later, rows))

    def test_the_trainer_refuses_it_without_a_prior(self) -> None:
        ids, segments, _ = data()
        with self.assertRaises(ValueError):
            loss(make_model("log-probs"), ids, segments, None)


class DepthEvidenceInputTest(unittest.TestCase):
    def test_other_parameters_and_the_random_stream_start_the_same(self) -> None:
        plain = dict(make_model("none").named_parameters())
        for kind in ("depth-evidence", "both"):
            model = make_model(kind)
            after = torch.rand(3)
            for name, p in plain.items():
                torch.testing.assert_close(dict(model.named_parameters())[name], p, rtol=0, atol=0, msg=f"{kind} {name}")
            make_model("none")
            torch.testing.assert_close(torch.rand(3), after, rtol=0, atol=0)

    def test_first_pass_is_unchanged(self) -> None:
        ids, segments, rows = data()
        ev = evidence_rows()
        inputs = {}
        for kind in ("none", "depth-evidence", "both"):
            model = make_model(kind)
            inputs[kind] = []
            model.rnn.register_forward_pre_hook(lambda _m, args, kind=kind: inputs[kind].append(args[0]))
            with torch.no_grad():
                inputs[kind].append(loss(model, ids, segments, rows, evidence=ev))
        for kind in ("depth-evidence", "both"):
            for a, b in zip(inputs["none"], inputs[kind], strict=True):
                torch.testing.assert_close(b, a, rtol=0, atol=0)

    def test_position_r_gets_evidence_row_r(self) -> None:
        ids, segments, rows = data()
        ev = evidence_rows()
        for kind in ("depth-evidence", "both"):
            model = make_model(kind)
            with torch.no_grad():
                model.prior_evidence_proj[1].weight.normal_()
                if model.prior_input_proj is not None:
                    model.prior_input_proj[1].weight.normal_()
            seen = []
            model.rnn.register_forward_pre_hook(lambda _m, args: seen.append(args[0].squeeze(0)))
            with torch.no_grad():
                loss(model, ids, segments[1:], rows[5:], offset=5, evidence=ev[5:])
                want = model.embed(ids[5:24]) + model.prior_evidence_proj(ev[5:24].reshape(19, -1))
                if model.prior_input_proj is not None:
                    want = want + model.prior_input_proj(rows[5:24])
            torch.testing.assert_close(torch.cat(seen), want, msg=kind)

    def test_training_moves_it_and_then_the_evidence_matters(self) -> None:
        ids, segments, rows = data()
        ev, other = evidence_rows(), evidence_rows(seed=9)
        model = make_model("depth-evidence")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            loss(model, ids, segments, rows, evidence=ev).backward()
            opt.step()
        self.assertGreater(model.prior_evidence_proj[1].weight.abs().sum().item(), 0.0)
        with torch.no_grad():
            self.assertNotAlmostEqual(
                loss(model, ids, segments, rows, evidence=ev).item(), loss(model, ids, segments, rows, evidence=other).item(), places=4
            )

    def test_refuses_without_evidence_or_width(self) -> None:
        ids, segments, rows = data()
        with self.assertRaises(ValueError):
            loss(make_model("depth-evidence"), ids, segments, rows)
        with self.assertRaises(ValueError):
            make_model("depth-evidence", width=0)


if __name__ == "__main__":
    unittest.main()
