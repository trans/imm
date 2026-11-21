# At the run's shapes (64 streams, ~700 queries, ~1,100 records, 4 heads of
# 48): does the fused kernel take the mask, what does an empty row give,
# and what does one forward and backward peak at.
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, T, M, D = 64, 4, 700, 1100, 48
q = torch.randn(B, H, T, D, device="cuda", requires_grad=True)
k = torch.randn(B, H, M, D, device="cuda", requires_grad=True)
m = torch.rand(B, 1, T, M, device="cuda") < 0.3
m[0, 0, 5] = False
torch.cuda.reset_peak_memory_stats()
try:
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(q, k, k, attn_mask=m)
        o.sum().backward()
    torch.cuda.synchronize()
    print(f"efficient kernel: ok; empty row nan={bool(torch.isnan(o[0, :, 5]).any())} "
          f"max={float(o[0, :, 5].nan_to_num().abs().max()):.3g}; peak {torch.cuda.max_memory_allocated() >> 20} MiB")
except RuntimeError as e:
    print("efficient kernel refused:", str(e)[:400])
