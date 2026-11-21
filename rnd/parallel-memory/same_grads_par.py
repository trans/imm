"""The batched path's loss and every gradient, old code (imm-speed, a353202)
against new (imm-par, parallel-memory): bit for bit, on the CPU, at two layouts with ragged
word segments, carried memory and dropout."""
import importlib.util
import sys

import numpy as np
import torch

S = "/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad"


def load(tree, name):
    sys.path.insert(0, f"{S}/{tree}")
    spec = importlib.util.spec_from_file_location(name, f"{S}/{tree}/scripts/run_segment_memory_model.py")
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    sys.path.pop(0)
    return m


old, new = load("imm-speed", "rsm_old"), load("imm-par", "rsm_new")
V, D, F = 40, 4, 11


def make(rsm, h, layers):
    torch.manual_seed(0)
    return rsm.SegmentMemoryLM(V, h, h, use_rope=True, rope_positions="char", memory_state="terminal", memory_record="written",
        rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn", attention_layers=layers, attention_heads=4,
        input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0, carry_bound="norm", carry_scale=32.0,
        prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=0.05, prior_residual_gate="context",
        prior_residual_gate_max=2.0, prior_residual_gate_features=F, prior_residual_state_features=[0, 3],
        skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
        record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0,
        utility_temperature=1.0, prior_input="both", prior_evidence_width=D * (V + 1), dropout=0.2)


rng = np.random.default_rng(1)
n = 3000
ids = torch.from_numpy(rng.integers(0, V, n + 1))
lengths = rng.integers(1, 9, n)
segs, pos = [], 0
while pos + lengths[pos] < n:
    segs.append((pos, pos + int(lengths[pos]))); pos += int(lengths[pos])
rows = dict(prior_log_probs=torch.log_softmax(torch.randn(n, V), -1).half(), prior_gate_features=torch.rand(n, F),
            prior_evidence=torch.rand(n, D, V + 1).half())

checked = 0
for h, layers, streams, k in ((32, 1, 8, 12), (32, 2, 16, 5)):
    models = [make(old, h, layers), make(new, h, layers)]
    models[1].load_state_dict(models[0].state_dict())
    stream_segments = old.split_streams(segs, streams)
    state = [(None, None), (None, None)]
    for u in range(3):  # three updates: memory and the carried state flow from one to the next
        batch = [s[u * k:(u + 1) * k] for s in stream_segments]
        spans = [(s[0][0], s[-1][1]) for s in batch]
        got = []
        for i, (rsm, m) in enumerate(zip((old, new), models)):
            m.train(); m.zero_grad(set_to_none=True)
            given = {kk: rsm.prior_rows_for_spans(v, spans, torch.device("cpu")) for kk, v in rows.items()}
            torch.manual_seed(100 + u)  # the same dropout draws
            loss, tokens, mem, hid, _ = m.segment_loss_streams(ids, batch, state[i][0], 16, carried_hidden=state[i][1],
                                                               carry_hidden=True, **given)
            (loss / tokens).backward()
            got.append((loss.detach().clone(), {nm: p.grad.clone() for nm, p in m.named_parameters() if p.grad is not None}))
            state[i] = ((mem[0].detach(), mem[1]), hid.detach())
        (la, ga), (lb, gb) = got
        assert torch.equal(la, lb), (h, layers, u, float(la), float(lb))
        assert ga.keys() == gb.keys()
        for nm in ga:
            assert torch.equal(ga[nm], gb[nm]), (h, layers, u, nm, float((ga[nm] - gb[nm]).abs().max()))
        checked += len(ga)
print(f"identical: loss and {checked} gradient tensors over 2 layouts x 3 updates")
