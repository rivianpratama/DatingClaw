#!/bin/bash
# Waits for e20r_lora_photo_full, then trains seeds 1 and 2 sequentially (one GPU job at a time).
cd /c/Users/Rivian/Documents/GitHub/BumbleClaw
R=/d/bumbleclaw_v6r1/results; L=/d/bumbleclaw_v6r1/logs
SNAP='D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935'
while [ ! -f $R/e20r_lora_photo_full/COMPLETE.json ]; do
  if ! tasklist //FI "IMAGENAME eq python.exe" | grep -q python && [ ! -f $R/e20r_lora_photo_full/COMPLETE.json ]; then echo "seed0 died; abort $(date)" >> $L/e20_queue.log; exit 1; fi
  sleep 120
done
for SEED in 1 2; do
  FREE=$(df -BG /d | tail -1 | awk '{print $4}' | tr -d G)
  if [ "$FREE" -lt 10 ]; then echo "disk low ${FREE}G; stop before seed $SEED $(date)" >> $L/e20_queue.log; exit 1; fi
  echo "start seed $SEED $(date)" >> $L/e20_queue.log
  PYTHONPATH=. TEMP='D:\BumbleTemp' /d/BumbleClawClipVenv/Scripts/python.exe -m bumbleclaw_v6.finetune.lora_siglip --snapshot "$SNAP" --side photo --full --run e20r_lora_photo_full_s$SEED --seed $SEED --epochs 7 --r 16 --alpha 32 --first-layer 20 > $L/e20r_lora_photo_full_s$SEED.log 2>&1
  echo "seed $SEED exit $? $(date)" >> $L/e20_queue.log
done
