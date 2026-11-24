# `rnd/` Chronology

The order in which the experiments under `rnd/` were tried. Each directory's
`README.md` has the idea, the method, the commands and the results. Numbers are
held-out perplexity per character unless a row says otherwise; each entry
names its own data and prior.

| Order | First seen | Experiment | Outcome |
| --- | --- | --- | --- |
| 1 | 2026-06-15 | [transition-categories](transition-categories/) | Inconclusive: an 8-dimensional SVD category code helped the 20k smoke (11.763 → 11.647) and hurt the Tiny tail split after one epoch (4.97976 → 5.10791). |
| 2 | 2026-06-15 | [edit-repair](edit-repair/) | Parked: fuzzy-node repair helps only on corrupted contexts (p=0.20: 4.464 → 4.377); the full-kernel results were invalid, as the gate never trained (fixed in 15efc86). |
| 3 | 2026-06-17 | [skip-horizons](skip-horizons/) | Inconclusive: learned horizon channels 10.056 against the baseline's 10.083 on the 50k Tiny slice (2,000-character eval); gate bias −2 hurts (10.294). |
| 4 | 2026-11-08 | [training-prior-leak](training-prior-leak/) | Fixed. Cross-fitting moves held-out perplexity by 0.3–0.5%. Strength 1.0 beats 0.75 by 6% on Tiny (3.7822 against 4.0144) and 2.9% on Gutenberg (3.4596 against 3.5619). |
| 5 | 2026-11-08 | [depth-evidence-input](depth-evidence-input/) | Positive: `--prior-input both` gains 1.04–1.28% on Tiny over 3 seeds (3.7341 against 3.7734); Gutenberg h256 best 3.2712 → 3.1767 with mobius's depth 16. |
| 6 | 2026-11-09 | [transformer-baseline-gutenberg5m](transformer-baseline-gutenberg5m/) | Reference: the nanoGPT character config (10.7M) 3.0212; 474k 3.2694 at 50k steps; imm then 3.4359, at about 3.6k characters/s against 1.9M. |
| 7 | 2026-11-09 | [batched-streams](batched-streams/) | Positive: 16 streams, 10 epochs, Tiny 3.6769 against 3.7734; Gutenberg 64 streams 3.3460 in 8.8 min against 3.4359 in about 65 min. |
| 8 | 2026-11-09 | [residual-width](residual-width/) | Positive: h256 at lr 1e-3 3.2711; h64 (179k) 3.1606 beats the 474k transformer by 3.3%; dropout 0.2 gives h256 3.0217 / 3.0395 (two seeds) against the 1.83M transformer's 3.0542. |
| 9 | 2026-11-09 | [word-segments](word-segments/) | Positive: Tiny 3-seed mean 3.6599 against 3.6782 (−0.50%); Gutenberg h256 depth 16 3.1767 → 3.1316 (−1.42%). |
| 10 | 2026-11-09 | [typo-noise](typo-noise/) | Negative: a typo costs imm about 14.0 / 13.8 / 13.3 bits, the same as Kneser–Ney; 1% input noise only regularises (3.1244). |
| 11 | 2026-11-10 | [mobius-prior-rows](mobius-prior-rows/) | Positive: mix_units −0.49% (3.0927); 19.7M characters of books in the prior −2.03% (3.0315); h256 with an 80-epoch cosine 2.9514 against 3.0076. |
| 12 | 2026-11-10 | [pg19-4.75m-split](pg19-4.75m-split/) | Reference: imm's prior 2.1183 bits/char; h256 3.7059 on it, 3.3220 on the 200M-book d16 prior; 1.0% behind the prior-in-input transformer (3.2887). |
| 13 | 2026-11-11 | [carry-bound](carry-bound/) | Bug fixed: one stream's carried norm ran from 9 to 3165 (held-out 121.1; 3.380 with tanh). The tanh run has no spike (3.3250) but trails by 0.5%; a norm cap of 32 adopted. |
| 14 | 2026-11-11 | [transformer-prior-pg19](transformer-prior-pg19/) | Reference: on 20M the 1.84M transformer plain 3.5689, prior on the logits 3.1904, plus the input 3.0987; at 400M 477k 3.1565, 1.84M 3.1578; d24 2.8131. Dropout-0.2 caveat. |
| 15 | 2026-11-12 | [prior-dropout](prior-dropout/) | Negative: without the prior imm's state is worth 9–10 perplexity; prior dropout 0.5 gets the own head to 3.6177 but costs 4.3% (3.2865 against 3.1517). |
| 16 | 2026-11-12 | [blocks-20m](blocks-20m/) | Mixed: 20M base 3.1517; a second block +0.40% (3.1392); 2 × h192 level (3.1466); parallel two blocks 3.1360; a third block only 0.03% (3.1351). |
| 17 | 2026-11-13 | [memory-knockouts](memory-knockouts/) | Mixed: memory off and retrained +2.1% / +1.0%, but zeroed +6.2%. At slot 63, zeroing costs the word GRU +12.3% and the attention +1.3%; retrained without each, +0.57% (2.7915) and +0.52% (2.7903). |
| 18 | 2026-11-13 | [parallel-memory](parallel-memory/) | Positive: 3.1518 against 3.1517, 3.2× faster in model time (276k against 77.6k characters/s). |
| 19 | 2026-11-14 | [word-tier](word-tier/) | Positive: word tier 3.1274 (−0.27%); own step 3.1155 at 400M and 3.1011 at 820M, 0.08% behind the 1.86M transformer (3.0987). |
| 20 | 2026-11-14 | [memory-window](memory-window/) | Positive: 128 records 2.7619 against 2.7738 (−0.43%) on the d24 word-tier model; 32 records +0.12%; 256 +0.22% against 128; on d16, 0.09%. |
| 21 | 2026-11-15 | [phrase-tier](phrase-tier/) | Negative: 3.1326 against 3.1274 (+0.17%); with own step 3.1140 against 3.1155 (−0.05% ±0.06). |
| 22 | 2026-11-16 | [learned-prior-strength](learned-prior-strength/) | Positive: `--prior-strength-gate` 3.0616 at 400M (−1.73%), 3.0563 at 820M (1.37% ahead of 3.0987). The no-prior row's 4.9% lead (3.3916) was later voided by the dropout finding. |
| 23 | 2026-11-16 | [imm-1.85m](imm-1.85m/) | Positive: 1,852,908 parameters, 3.0575 on d16 (−1.34% against the transformer); 2.7757 on d24 (−1.33% against the d24 transformer's 2.8131). |
| 24 | 2026-11-17 | [topic-tiers](topic-tiers/) | Negative: by piece R0 2.7735, R1 2.7738, R2 2.7785, R2c 2.7746, R3 2.7867, S1 2.7751, S2 2.7736. Sentence swap +0.20%; a post-hoc topic prior −1.01%. The user closed the line. |
| 25 | 2026-11-18 | [data-context-size](data-context-size/) | Mixed: 62% of the loss is a word's first two letters. Half the data +2.46% (2.8299), 256 records +0.22%, h256 +0.79%: data binds. |
| 26 | 2026-11-19 | [no-prior-pair-1.8m](no-prior-pair-1.8m/) | No prior, both at dropout 0: the 1.84M transformer leads imm (1.51M trained), 3.2060 / 3.1862 against 3.2862, imm +2.50% / +3.14%. The first reading, imm −5.20%, came from the transformer's dropout 0.2. |
| 27 | 2026-11-19 | [char-records](char-records/) | Positive: character records close imm's copying gap. 256 records 3.1519 (−1.07% against the block-1,024 transformer), 1,024 records 3.0440 (−4.46%). Reading them in runs was no faster and was dropped. |
| 28 | 2026-11-19 | [scale-10.8m](scale-10.8m/) | At ~10.8M the transformer leads: 2.7458 against imm's 2.8188 (+2.66%). imm gains 7.74% from 1.59M, the transformer 13.8%. |
| 29 | 2026-11-20 | [rope-far-positions](rope-far-positions/) | Bug fixed: RoPE angles lost precision far into the text (cac43fb). Fixed code in bf16 scores 3.0552 against 3.0440 (+0.37% ±0.10, no measurable cost), 1.67× faster. |
| 30 | 2026-11-20 | [architecture-1.59m](architecture-1.59m/) | Against slot 127's 3.0552: depth records 3.0109 (−1.45%), the GRU hierarchy alone 3.3473 (+9.56%), char-kv 3.0449 (−0.34%). |
| 31 | 2026-11-20 | [half-lr-10.8m](half-lr-10.8m/) | Negative: half the learning rate at 10.8M (queued as 138, ran as 145 on 2026-11-21): 2.8230, +0.15% ±0.11 against 2.8188. 1e-3 stays. |
| 32 | 2026-11-20 | [simplified-no-words](simplified-no-words/) | No word records or word tier: 3.0705, +0.50% ±0.11 against 3.0552. The word level is worth ~0.5% at 1.59M. |
| 33 | 2026-11-20 | [depth-records-10.8m](depth-records-10.8m/) | Positive: depth records at 10.8M 2.8004, −0.65% ±0.11 against 2.8188. imm now trails the 10.8M transformer by 1.99%. |
| 34 | 2026-11-20 | [attention-only-words](attention-only-words/) | Negative: attention alone (no GRU) scores 3.423 at epoch 3, 5.72 points behind slot 127. Mean word records lose at every cutoff: 0.68, 0.89 (rotated), 1.35 and 6.77 points behind A. |
| 35 | 2026-11-20 | [reading-early](reading-early/) | Reference: in 8 pairs, gaps of about 1 point or more at epoch 3 kept their sign; smaller ones moved up to ~0.5. Learning rate and dropout need full runs. |
| 36 | 2026-11-21 | [gru-tree](gru-tree/) | Negative (slot 147, epoch 3): the 10-level tree 4.016 against a plain GRU of the same size 3.664 (+9.61%); only levels 1–2 open their gates. |
| 37 | 2026-11-21 | [two-char-updates](two-char-updates/) | Stage 1: a tiny character GRU updated every 2 characters scores 7.473 against 6.012 with a 256 window (+24.3%), like counting 2–3 characters back. Stage 2: level 1 on top, on its own two steps, makes it worse (8.165, +9.3%; +14.6% against the same size), though its gate opens. Stages 3–4: with one clock for both, level 1 still harms (6.866 against 6.523 alone, +5.2%), attached or not. |
| 38 | 2026-11-21 | [pair-level](pair-level/) | Mixed: level 1 making the prediction through a small decoder per pair loses to a character GRU of the same size by about a fifth (7.975 against 6.523 every 4 characters, 7.261 against 6.012 every 256), mostly on each pair's second character. A wider decoder helps most; at the GRU's compute it still loses, 6.753 against 6.012 (+12.3%). In equal wall-clock time on CPU it can be seven times larger and then wins a little, 5.753 against a GRU 154's 5.820 (−1.2%). |
| 39 | 2026-11-22 | [depth-width-10.8m](depth-width-10.8m/) | Parked: 6 blocks at width 344 (same parameters as slot 137's 3 at 448) fit the GPU only at a 64 x 96 cap, 3.2x slower an update, and scoring doesn't fit, so nothing trained. The check before it: at epoch 8 imm's training loss is 0.0118 nats above the transformer's and its held-out 0.0196, so it mostly fits less, not overfits more. |
| 40 | 2026-11-24 | [fm-index-1gb](fm-index-1gb/) | Positive (CPU, the second machine): an FM-index of 1 GB of books gives the follower histograms a prior needs, depth 1–24, in 15.2 µs a held-out character on one core with plain bit vectors (851 MB), or 179 µs with RRR (250 MB). Built in 2.2 min at 4.8 GB. About 1.2x per doubling of the text. 10 GB in one piece needs another suffix-array build. |
