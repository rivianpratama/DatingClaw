#!/bin/bash
# After E22 seed0: (1) v6r5 prod GPU parity+latency FIRST, (2) LoRA single-seed shadow release + parity/latency.
cd /c/Users/Rivian/Documents/GitHub/BumbleClaw
export PYTHONPATH=. TEMP='D:\BumbleTemp'; PY=/d/BumbleClawClipVenv/Scripts/python.exe
L=/d/bumbleclaw_v6r1/logs/post_seed0.log; R=/d/bumbleclaw_v6r1/results
while [ ! -f $R/e22r_lora_photo_box_s0/COMPLETE.json ]; do
  grep -q "seed0 exit" /d/bumbleclaw_v6r1/logs/e22_queue.log && [ ! -f $R/e22r_lora_photo_box_s0/COMPLETE.json ] && { echo "seed0 failed $(date)" >> $L; break; }
  sleep 60
done
echo "v6r5 parity start $(date)" >> $L
$PY scratch/release_parity_v6r5.py > /d/bumbleclaw_v6r1/logs/e24_v6r5_parity.log 2>&1; echo "v6r5 parity exit $? $(date)" >> $L
IMGS=$($PY -c "
import numpy as np
from bumbleclaw_v6.data.load import load_snapshot
m=load_snapshot(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935')['samples']
r=np.random.default_rng(1).choice(len(m),120,replace=False); print(' '.join(m['path'].iloc[r].astype(str)))" 2>/dev/null)
$PY -m bumbleclaw_v6.runtime.benchmark 'D:\bumbleclaw_v6r1\releases\v6r5_photo_tight_wide_blend' $IMGS > $R/e24_v6r5_latency.json 2>/dev/null; echo "v6r5 latency exit $? $(date)" >> $L
[ -f $R/e22r_lora_photo_box_s0/COMPLETE.json ] || exit 0
$PY -m bumbleclaw_v6.runtime.lora_scorer build --runs $R/e22r_lora_photo_box_s0 --out 'D:\bumbleclaw_v6r1\releases\v6r6_lora_photo_box_s0' --name v6r6_lora_photo_box_s0 > /d/bumbleclaw_v6r1/logs/e25_lora_build.log 2>&1; echo "lora build exit $? $(date)" >> $L
$PY scratch/e25_lora_parity_latency.py > /d/bumbleclaw_v6r1/logs/e25_lora_parity.log 2>&1; echo "lora parity exit $? $(date)" >> $L
