# Trained parameters of the residual (those that get a gradient) for a given
# width and block count, with the 20M run's settings: 96 characters, depth-16
# evidence (16 x 97), 11 gate features, 2 state features, prior input both.
import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec); sys.modules["rsm"] = rsm; spec.loader.exec_module(rsm)
V, D, F = 96, 16, 11
def count(h, layers, mix="gated-xattn"):
    m = rsm.SegmentMemoryLM(V, h, h, use_rope=True, rope_positions="char", memory_state="terminal", memory_record="written",
        rnn_core="gru", attn_interface_norm="none", mixing=mix, attention_layers=layers, attention_heads=4,
        input_rope=False, feedback_gate_bias=-1.0, feedback_delta_cap=1.0, prior_strength=1.0, prior_residual_scale=1.0,
        prior_residual_l2=0.05, prior_residual_gate="context", prior_residual_gate_max=2.0, prior_residual_gate_features=F,
        prior_residual_state_features=[0, 3], skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0,
        skip_prior_channels=0, record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0,
        utility_aux_weight=0.0, utility_temperature=1.0, prior_input="both", prior_evidence_width=D * (V + 1), dropout=0.2)
    for p in m.parameters():
        if p.abs().sum() == 0: p.data.normal_(0, 0.1)
    n = 200; ids = torch.randint(0, V, (n + 1,))
    rows = dict(prior_log_probs=torch.log_softmax(torch.randn(n, V), -1), prior_gate_features=torch.randn(n, F),
                prior_evidence=torch.rand(n, D, V + 1))
    segs = [(i, i + 5) for i in range(0, n, 5)]
    streams = rsm.split_streams(segs, 2)
    spans = [(s[0][0], s[-1][1]) for s in streams]
    m.train()
    loss, *_ = m.segment_loss_streams(ids, streams, None, 8, carry_hidden=True,
        **{k: rsm.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in rows.items()})
    loss.backward()
    return sum(p.numel() for p in m.parameters() if p.grad is not None)
for h, layers, mix in ((416, 1, "none"), (432, 1, "none"), (440, 1, "none")):
    print(f"h{h} blocks={layers} mixing={mix}: {count(h, layers, mix):,}")
