#!/bin/sh
# Waits for imm's slot $1, runs the four knockout scorings (run.sh), ends the slot.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
SLOTS="sh /home/trans/Projects/air/tools/slots"
renice -n 19 -p $$ >/dev/null; ionice -c 3 -p $$
sid=$1
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1; fi
  sleep 20
done
echo "slot $sid granted $(date -Is)"
sh $S/ko/run.sh
$SLOTS done @imm
echo "slot ended $(date -Is)"
