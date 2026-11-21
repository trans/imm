# Written 2026-11-21 for this record (not part of the original analysis).
# The per-epoch gap between two runs, from their CSVs in runs/
# (columns: epoch, step, train_nll, val_nll, val_ppl, ...), three ways:
#   points    100 x (val_nll_A - val_nll_B), nats per character
#   pct_nll   100 x (ln ppl_A / ln ppl_B - 1), from the 3-decimal
#             perplexities the logs print: the unit of the 2026-11-20 table
#   pct_ppl   100 x (ppl_A / ppl_B - 1)
# Usage, from runs/:
#   awk -f gaps_by_epoch.awk A_ID_A_ID.csv B_ID_B_ID.csv
BEGIN { FS = "," }
FNR == 1 { f++ }
FNR > 1 && $1 > 0 { nll[f, $1] = $4; ppl[f, $1] = $5 }
END {
  split("1 2 3 4 8", E, " ")
  for (i = 1; i <= 5; i++) {
    e = E[i]
    if (!((1, e) in nll) || !((2, e) in nll)) continue
    a3 = sprintf("%.3f", ppl[1, e]); b3 = sprintf("%.3f", ppl[2, e])
    printf "e%d  points %+.2f   pct_nll %+.2f   pct_ppl %+.2f\n", e,
      100 * (nll[1, e] - nll[2, e]), 100 * (log(a3) / log(b3) - 1),
      100 * (ppl[1, e] / ppl[2, e] - 1)
  }
}
