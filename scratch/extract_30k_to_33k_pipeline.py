import os
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    python_clip = r"D:\BumbleClawClipVenv\Scripts\python.exe"

    ref_store = root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"
    clip_store = root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"
    siglip_store = root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"

    # 1. CLIP ViT-B/32
    if not clip_store.exists():
        print(f"Extracting CLIP ViT-B/32 for {ref_store}...")
        cmd_clip = [
            python_clip, "build_clip_store.py",
            "--store", str(ref_store),
            "--output", str(clip_store),
            "--batch-size", "64",
            "--provider", "cuda"
        ]
        subprocess.run(cmd_clip, check=True)
        print(f"[OK] Saved {clip_store}")
    else:
        print(f"[EXISTS] {clip_store}")

    # 2. SigLIP-384
    if not siglip_store.exists():
        print(f"Extracting SigLIP-384 for {ref_store}...")
        cmd_siglip = [
            python_clip, "build_siglip_store.py",
            "--store", str(ref_store),
            "--output", str(siglip_store),
            "--batch-size", "32",
            "--workers", "8"
        ]
        subprocess.run(cmd_siglip, check=True)
        print(f"[OK] Saved {siglip_store}")
    else:
        print(f"[EXISTS] {siglip_store}")

    # 2.5 Train 10k SigLIP if missing
    ckpt_10k = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_10k.pt"
    if not ckpt_10k.exists():
        print("Training 10k SigLIP sub-center...")
        subprocess.run([python_clip, "scratch/train_subcenter_siglip_10k.py"], check=True)

    # 3. Run Benchmark
    print("\nRunning Master Tournament Benchmark...")
    cmd_bench = [
        python_clip, "scratch/run_benchmark_10k_20k_30k_on_33k_holdout.py"
    ]
    subprocess.run(cmd_bench, check=True)

if __name__ == "__main__":
    main()
