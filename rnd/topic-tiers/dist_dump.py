"""The held-out text's full next-character distributions (for @air's
post-hoc topic-prior test, the user via @air 0ffdfd38, 2026-11-18). Runs
run_segment_memory_model.py's main() from the current directory's repo, with
write_char_losses wrapped: while it scores the held-out text, each call of
F.cross_entropy(..., reduction="none") (the stats path's per-character
losses, which loss_sink keeps next) also keeps log_softmax of its logits.
Each kept block is checked against the losses the sink keeps for it: the
target's -log-prob must equal its loss.

Writes DIST_OUT (npz, uncompressed): log_probs [N, V] float16, row r scoring
target text[r + 1] (positions 0..N-1, checked), columns in the model's
vocabulary order (checked against VOCAB_TXT if given); positions int64;
losses float32 (as char_losses); text int16; vocab.

    DIST_OUT=out.npz VOCAB_TXT=vocab.txt python dist_dump.py <scorer args, with --char-loss-output>
"""
import importlib.util, os, sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.getcwd()
spec = importlib.util.spec_from_file_location("rsm", f"{REPO}/scripts/run_segment_memory_model.py")
M = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = M
spec.loader.exec_module(M)
out = os.environ["DIST_OUT"]
vocab_txt = os.environ.get("VOCAB_TXT")
orig_write = M.write_char_losses
orig_ce = F.cross_entropy


def wrapped(model, args, eval_ids, eval_segments, eval_input_ids, vocab, *a, **k):
    blocks = []

    def ce(logits, targets, *ca, **ck):
        loss = orig_ce(logits, targets, *ca, **ck)
        if ck.get("reduction") == "none" and logits.dim() == 2:
            lp = torch.log_softmax(logits.detach().float(), dim=-1)
            blocks.append((lp.to("cpu", torch.float16), lp.gather(1, targets[:, None])[:, 0].cpu(), targets.cpu()))
            holder["sink"] = model.loss_sink  # write_char_losses' own list, which the next line of the model fills
        return loss

    holder = {}
    M.F.cross_entropy = ce
    try:
        result = orig_write(model, args, eval_ids, eval_segments, eval_input_ids, vocab, *a, **k)
    finally:
        M.F.cross_entropy = orig_ce
    sink = holder.get("sink") or []
    if len(blocks) != len(sink):
        raise RuntimeError(f"dist_dump: {len(blocks)} cross_entropy blocks against {len(sink)} loss blocks")
    for (lp, tl, _), (pos, loss) in zip(blocks, sink):
        if lp.shape[0] != pos.shape[0] or not torch.allclose(-tl, loss, atol=1e-4):
            raise RuntimeError("dist_dump: a kept block doesn't match its losses")
    pos = torch.cat([p for p, _ in sink]).numpy()
    losses = torch.cat([x for _, x in sink]).numpy()
    lp = torch.cat([b[0] for b in blocks]).numpy()
    order = np.argsort(pos, kind="stable")
    pos, losses, lp = pos[order], losses[order], lp[order]
    if not np.array_equal(pos, np.arange(len(pos))):
        raise RuntimeError("dist_dump: positions aren't 0..N-1")
    chars = list(vocab.chars)
    if vocab_txt:
        want = list(open(vocab_txt, encoding="utf-8").read())
        if want and want[-1] == "\n" and len(want) == len(chars) + 1:
            want = want[:-1]
        if want != chars:
            raise RuntimeError(f"dist_dump: the model's vocabulary isn't {vocab_txt}'s order")
    text = eval_ids.cpu().numpy().astype(np.int64)
    # float16 rounding: the target's log-prob as stored, against the loss
    back = -lp[np.arange(len(pos)), text[pos + 1]].astype(np.float64)
    mass = np.exp(lp.astype(np.float64)).sum(axis=1)
    np.savez(out, log_probs=lp, positions=pos.astype(np.int64), losses=losses.astype(np.float32),
             text=text.astype(np.int16), vocab=np.array(chars))
    print(f"dist_dump n={len(pos)} V={lp.shape[1]} mean_loss={losses.astype(np.float64).mean():.6f} "
          f"ppl={np.exp(losses.astype(np.float64).mean()):.4f} fp16_mean_loss={back.mean():.6f} "
          f"max_abs_fp16_err={np.abs(back - losses).max():.5f} row_mass={mass.min():.4f}..{mass.max():.4f} -> {out}", flush=True)
    return result


M.write_char_losses = wrapped
sys.argv = [f"{REPO}/scripts/run_segment_memory_model.py"] + sys.argv[1:]
M.main()
