#!/bin/sh
# The user, to air (2026-11-20 ~04:55 UTC), paraphrased: let imm's current
# run finish, then max's runs, then imm's again. Slot 126's script
# (s10/imm10e_slot.sh) goes on from the 10.8M run and its scoring to the
# 1.59M rerun; when that run starts, stop it, so the script's own failure
# path (train || finish 1) ends the slot. The rerun waits in its own slot
# (s10/imm159_slot.sh), behind max's.
renice -n 19 -p $$ >/dev/null
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
log=$S/s10/slot_126.log
pat="^python -B scripts/run_segment_memory_model.py .*--run-id pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_rel_bf16_e8 --output"
until grep -q "^start imm pg19s100m_h192_" $log || grep -q "^slot ended" $log; do sleep 5; done
if grep -q "^slot ended" $log; then echo "slot 126 ended on its own $(date -Is)"; exit 0; fi
i=0
until pkill -f -- "$pat"; do
  i=$((i + 1)); [ $i -gt 120 ] && { echo "the 1.59M rerun never appeared $(date -Is)"; exit 1; }
  sleep 1
done
echo "stopped the 1.59M rerun at its start $(date -Is)"
until grep -q "^slot ended" $log; do sleep 5; done
echo "slot 126 ended $(date -Is)"
