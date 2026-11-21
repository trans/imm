"""Parameters a checkpoint's optimizer updated (AdamW keeps state only for
parameters that got a gradient): python used_params.py CKPT [CKPT ...]"""
import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    o = ck["optimizer_state_dict"]; st = o["state"]
    sd = ck["model_state_dict"]
    print(f"{sum(s['exp_avg'].numel() for s in st.values()):>10,} used of {sum(v.numel() for v in sd.values()):>11,}  {p.split('/')[-1][:80]}")
