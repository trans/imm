#!/bin/sh
# Waits until air's slots tool shows slot 7 granted to @imm, then runs the
# pair (h256 x2 blocks, then h128 x3) and ends imm's slot. Starts nothing
# heavy before the grant.
renice -n 19 -p $$ >/dev/null
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
SLOTS="sh /home/trans/Projects/air/tools/slots"
while ! $SLOTS list 2>/dev/null | grep -q '^in use: 7 @imm,'; do
  if ! $SLOTS list 2>/dev/null | grep -qE '^(in use: |[0-9]+\. )7 @imm,'; then
    echo "slot 7 is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 30
done
echo "slot 7 granted $(date -Is)"
sh $S/pg19_20m_slot.sh pair
$SLOTS done @imm
echo "slot ended $(date -Is)"
