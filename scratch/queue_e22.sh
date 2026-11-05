#!/bin/bash
# E22: runtime boxes -> verify -> single-seed photo-only LoRA on corrected boxes.
cd /c/Users/Rivian/Documents/GitHub/BumbleClaw
L=/d/bumbleclaw_v6r1/logs/e22_queue.log; F=/d/bumbleclaw_v6r1/features
SNAP='D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935'
export PYTHONPATH=. TEMP='D:\BumbleTemp'
PY=/d/BumbleClawClipVenv/Scripts/python.exe
echo "boxes start $(date)" >> $L
$PY -m bumbleclaw_v6.features.boxes --snapshot "$SNAP" --out $F/boxes_runtime_v1.npz > /d/bumbleclaw_v6r1/logs/e22_boxes.log 2>&1 || { echo "boxes FAILED $(date)" >> $L; exit 1; }
$PY scratch/verify_boxes.py >> $L 2>&1 || { echo "box verification FAILED $(date)" >> $L; exit 1; }
echo "seed0 start $(date)" >> $L
$PY -m bumbleclaw_v6.finetune.lora_siglip --snapshot "$SNAP" --side photo --full --boxes 'D:\bumbleclaw_v6r1\features\boxes_runtime_v1.npz' --run e22r_lora_photo_box_s0 --seed 0 --epochs 7 --r 16 --alpha 32 --first-layer 20 > /d/bumbleclaw_v6r1/logs/e22r_lora_photo_box_s0.log 2>&1
echo "seed0 exit $? $(date)" >> $L
