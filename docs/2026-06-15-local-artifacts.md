# 2026-06-15 Local Artifact Handoff

The AGPT-Ultra handoff artifacts needed for immediate IMM reproduction were
copied into this workspace.

## Copied Data

```text
data/gutenberg_5m.txt
data/.splits/4bfd5a43d446644a/
data/input.txt
data/.splits/2b7ded401e96b610/
data/.splits/4fa9aec1db6b3aea/
data/.splits/30e6aec9b3d3278b/
data/.splits/d60be5c5bf1205f6/
```

The Gutenberg split manifest records the original source SHA256:

```text
bebb2d90b3a6486aa3c49ec78f11e95668f509530a3f97513c38a5876f6a5ced
```

The Tiny Shakespeare source SHA256 is:

```text
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

## Copied Runs

```text
runs/gutenberg_count_gate_d8_entropy_suffix.json
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt.train_log_probs.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt.train_gate_features.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt.eval_log_probs.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt.eval_gate_features.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_200k.pt
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_200k.pt.train_log_probs.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_200k.pt.train_gate_features.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_200k.pt.eval_log_probs.dat
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_200k.pt.eval_gate_features.dat
runs/gutenberg5m_priorstrength075_residual025_l2_005_e1_gutenberg5m_priorstrength075_residual025_l2_005_e1.csv
runs/gutenberg5m_priorstrength075_residual025_l2_005_e1_gutenberg5m_priorstrength075_residual025_l2_005_e1.pt
runs/gutenberg5m_priorstrength075_residual025_l2_005_e1_gutenberg5m_priorstrength075_residual025_l2_005_e1.best.pt
runs/priorstrength075_residual025_l2_005_200k_e1_priorstrength075_residual025_l2_005_200k_e1.csv
runs/priorstrength075_residual025_l2_005_200k_e3_resume_priorstrength075_residual025_l2_005_200k_e3_resume.csv
runs/priorstrength075_residual025_l2_005_200k_e5_resume_priorstrength075_residual025_l2_005_200k_e5_resume.csv
runs/tiny_shakespeare_smoke_count_prior.pt
runs/tiny_shakespeare_smoke_count_prior.pt.train_log_probs.dat
runs/tiny_shakespeare_smoke_count_prior.pt.train_gate_features.dat
runs/tiny_shakespeare_smoke_count_prior.pt.eval_log_probs.dat
runs/tiny_shakespeare_smoke_count_prior.pt.eval_gate_features.dat
runs/tiny_shakespeare_smoke_tiny_shakespeare_smoke.csv
runs/tiny_shakespeare_smoke_tiny_shakespeare_smoke.best.pt
runs/tiny_shakespeare_smoke_reload_tiny_shakespeare_smoke_reload.csv
runs/tiny_shakespeare_smoke_reload_tiny_shakespeare_smoke_reload.best.pt
runs/tiny_shakespeare_train_smoke_e2_count_prior.pt
runs/tiny_shakespeare_train_smoke_e2_count_prior.pt.train_log_probs.dat
runs/tiny_shakespeare_train_smoke_e2_count_prior.pt.train_gate_features.dat
runs/tiny_shakespeare_train_smoke_e2_count_prior.pt.eval_log_probs.dat
runs/tiny_shakespeare_train_smoke_e2_count_prior.pt.eval_gate_features.dat
runs/tiny_shakespeare_train_smoke_e2_tiny_shakespeare_train_smoke_e2.csv
runs/tiny_shakespeare_train_smoke_e2_tiny_shakespeare_train_smoke_e2.pt
runs/tiny_shakespeare_train_smoke_e2_tiny_shakespeare_train_smoke_e2.best.pt
runs/tiny_shakespeare_transition_category_e2_tiny_shakespeare_transition_category_e2.csv
runs/tiny_shakespeare_transition_category_e2_tiny_shakespeare_transition_category_e2.pt
runs/tiny_shakespeare_transition_category_e2_tiny_shakespeare_transition_category_e2.best.pt
runs/tiny_shakespeare_full_count_prior.pt
runs/tiny_shakespeare_full_count_prior.pt.train_log_probs.dat
runs/tiny_shakespeare_full_count_prior.pt.train_gate_features.dat
runs/tiny_shakespeare_full_count_prior.pt.eval_log_probs.dat
runs/tiny_shakespeare_full_count_prior.pt.eval_gate_features.dat
runs/tiny_shakespeare_full_count_prior.pt.train_gate_features.transition_category_svd_d8.dat
runs/tiny_shakespeare_full_baseline_e1_tiny_shakespeare_full_baseline_e1.csv
runs/tiny_shakespeare_full_baseline_e1_tiny_shakespeare_full_baseline_e1.pt
runs/tiny_shakespeare_full_baseline_e1_tiny_shakespeare_full_baseline_e1.best.pt
runs/tiny_shakespeare_full_transition_category_e1_tiny_shakespeare_full_transition_category_e1.csv
runs/tiny_shakespeare_full_transition_category_e1_tiny_shakespeare_full_transition_category_e1.pt
runs/tiny_shakespeare_full_transition_category_e1_tiny_shakespeare_full_transition_category_e1.best.pt
```

The mmap cache `.pt` files use relative sidecar paths, so they are self-contained
as long as the sidecar files stay next to the `.pt` files.

## Known Validation

```sh
python -m unittest discover -s tests
```

## Experiment Configs

The long commands below are available as JSON configs. Command-line flags can
override config values.

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_smoke.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_train_smoke_e2.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_transition_category_e2.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_baseline_e1.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_transition_category_e1.json
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_priorstrength075_residual025_l2_005_e1.json
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_priorstrength075_residual025_l2_005_e2_resume.json
```

## Tiny Shakespeare Smoke

This bounded command exercises the same count-prior plus residual LMA path on a
50k/5k Tiny Shakespeare slice. It should finish in a few seconds on CPU. The
first run builds `runs/tiny_shakespeare_smoke_count_prior.pt`; subsequent runs
should report `cache_loaded=True`.

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_smoke.json
```

Equivalent expanded command:

```sh
python scripts/run_segment_memory_model.py \
  --run-id tiny_shakespeare_smoke \
  --input data/input.txt \
  --vocab-input data/input.txt \
  --max-train-chars 50000 \
  --eval-max-chars 5000 \
  --epochs 0 \
  --eval-initial \
  --mixing gated-xattn \
  --carry-hidden \
  --memory-record written \
  --rope-positions char \
  --terminal-record-aux-weight 0.0 \
  --max-memory 8 \
  --hidden-size 32 \
  --embedding-size 32 \
  --attention-heads 4 \
  --attention-layers 1 \
  --lr 0.0003 \
  --segments-per-update 32 \
  --eval-every 0 \
  --prior-strength 0.75 \
  --prior-residual-scale 0.25 \
  --prior-residual-l2 0.05 \
  --prior-residual-gate context \
  --prior-residual-state-features 0,3 \
  --count-prior frozen \
  --count-prior-depth 8 \
  --count-prior-epochs 5 \
  --count-prior-max-fit-positions 2000 \
  --count-prior-extra-features entropy_delta,suffix_stats \
  --count-prior-impl packed \
  --count-prior-precompute numpy \
  --count-prior-storage mmap \
  --count-prior-log-prob-dtype float16 \
  --count-prior-cache runs/tiny_shakespeare_smoke_count_prior.pt \
  --output runs/tiny_shakespeare_smoke.csv
```

Observed local smoke result:

```text
cache_loaded=True
val_ppl=9.656648997764249
runtime_sec=2.35
```

## Tiny Shakespeare Train Smoke

This short run trains for two epochs on a 20k/2k Tiny Shakespeare slice. It is
the fastest check that the train loop, epoch-end eval, checkpointing, and cache
path all work together.

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_train_smoke_e2.json
```

Observed local result:

```text
epoch 0 val_ppl=12.137462358431456
epoch 1 val_ppl=12.044675912756064
epoch 2 val_ppl=11.763123904345457
runtime_sec=31.54
```

## Tiny Shakespeare Transition Category

This is the first implementation of distributional categories induced by common
successor sets. It appends eight SVD transition-category features to the
count-prior feature rows and feeds them to the residual-state MLP.

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_transition_category_e2.json
```

Observed local comparison:

```text
baseline epoch 2 val_ppl=11.763123904345457
category epoch 2 val_ppl=11.64748002343055
```

## Full Tiny Shakespeare Category Check

The same category experiment was run on the full Tiny Shakespeare 95/5 tail
split for one epoch.

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_baseline_e1.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_transition_category_e1.json
```

Observed local comparison:

```text
baseline epoch 1 val_ppl=4.979760983655813
category epoch 1 val_ppl=5.107905667704909
```

The full split did not confirm the small-slice gain.

## Full Gutenberg Reproduction

This command should reproduce the current best one-epoch result from the local
copy of the artifacts.

```sh
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_priorstrength075_residual025_l2_005_e1.json
```

Equivalent expanded command:

```sh
python scripts/run_segment_memory_model.py \
  --run-id gutenberg5m_priorstrength075_residual025_l2_005_e1_repro \
  --input data/gutenberg_5m.txt \
  --train-input data/.splits/4bfd5a43d446644a/train_corpus.txt \
  --eval-input data/.splits/4bfd5a43d446644a/heldout_corpus.txt \
  --vocab-input data/gutenberg_5m.txt \
  --eval-max-chars 250000 \
  --epochs 1 \
  --eval-initial \
  --mixing gated-xattn \
  --carry-hidden \
  --memory-record written \
  --rope-positions char \
  --terminal-record-aux-weight 0.0 \
  --max-memory 16 \
  --hidden-size 64 \
  --embedding-size 64 \
  --attention-heads 4 \
  --attention-layers 1 \
  --lr 0.0003 \
  --segments-per-update 32 \
  --eval-every 0 \
  --prior-strength 0.75 \
  --prior-residual-scale 0.25 \
  --prior-residual-l2 0.05 \
  --prior-residual-gate context \
  --prior-residual-state-features 0,3 \
  --count-prior frozen \
  --count-prior-depth 8 \
  --count-prior-epochs 100 \
  --count-prior-max-fit-positions 50000 \
  --count-prior-extra-features entropy_delta,suffix_stats \
  --count-prior-impl packed \
  --count-prior-theta-json runs/gutenberg_count_gate_d8_entropy_suffix.json \
  --count-prior-precompute numpy \
  --count-prior-storage mmap \
  --count-prior-log-prob-dtype float16 \
  --count-prior-cache runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt \
  --profile \
  --progress-every-steps 5000 \
  --output runs/gutenberg5m_priorstrength075_residual025_l2_005_e1_repro.csv
```

## Full Epoch 2 Resume

```sh
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_priorstrength075_residual025_l2_005_e2_resume.json
```

Equivalent expanded command:

```sh
python scripts/run_segment_memory_model.py \
  --resume-checkpoint runs/gutenberg5m_priorstrength075_residual025_l2_005_e1_gutenberg5m_priorstrength075_residual025_l2_005_e1.pt \
  --run-id gutenberg5m_priorstrength075_residual025_l2_005_e2_resume \
  --input data/gutenberg_5m.txt \
  --train-input data/.splits/4bfd5a43d446644a/train_corpus.txt \
  --eval-input data/.splits/4bfd5a43d446644a/heldout_corpus.txt \
  --vocab-input data/gutenberg_5m.txt \
  --eval-max-chars 250000 \
  --epochs 2 \
  --mixing gated-xattn \
  --carry-hidden \
  --memory-record written \
  --rope-positions char \
  --terminal-record-aux-weight 0.0 \
  --max-memory 16 \
  --hidden-size 64 \
  --embedding-size 64 \
  --attention-heads 4 \
  --attention-layers 1 \
  --lr 0.0003 \
  --segments-per-update 32 \
  --eval-every 0 \
  --prior-strength 0.75 \
  --prior-residual-scale 0.25 \
  --prior-residual-l2 0.05 \
  --prior-residual-gate context \
  --prior-residual-state-features 0,3 \
  --count-prior frozen \
  --count-prior-depth 8 \
  --count-prior-epochs 100 \
  --count-prior-max-fit-positions 50000 \
  --count-prior-extra-features entropy_delta,suffix_stats \
  --count-prior-impl packed \
  --count-prior-theta-json runs/gutenberg_count_gate_d8_entropy_suffix.json \
  --count-prior-precompute numpy \
  --count-prior-storage mmap \
  --count-prior-log-prob-dtype float16 \
  --count-prior-cache runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt \
  --profile \
  --progress-every-steps 5000 \
  --output runs/gutenberg5m_priorstrength075_residual025_l2_005_e2_resume.csv
```

Use `--eval-every 0` for epoch-end evaluation. `--eval-every 1` evaluates every
optimizer update and is prohibitively slow for these runs.
