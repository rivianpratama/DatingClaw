import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    python_clip = r"D:\BumbleClawClipVenv\Scripts\python.exe"
    
    print("=" * 80)
    print("PHASE 2 MASTER PIPELINE AUTOMATION")
    print("=" * 80)

    # 1. Check if 20k SigLIP store is ready
    sig_20k_path = root / "embeddings" / "siglip_store_bumble_20k_pure.npz"
    if not sig_20k_path.exists():
        print(f"Waiting for {sig_20k_path} to complete extraction...")
        while not sig_20k_path.exists():
            time.sleep(10)
    print(f"[OK] Found {sig_20k_path}")

    # 2. Extract 22k-to-30k SigLIP store if missing
    sig_tail_path = root / "embeddings" / "siglip_store_bumble_22k_to_30k.npz"
    ref_tail_path = root / "embeddings" / "reference_store_bumble_22k_to_30k.npz"
    if not sig_tail_path.exists():
        print(f"\nExtracting SigLIP embeddings for 22k-30k ({ref_tail_path})...")
        cmd = [
            python_clip, "build_siglip_store.py",
            "--store", str(ref_tail_path),
            "--output", str(sig_tail_path),
            "--batch-size", "32",
            "--workers", "8"
        ]
        subprocess.run(cmd, check=True)
    print(f"[OK] Found {sig_tail_path}")

    # 3. Run Phase 2 Training & Benchmark
    print("\nRunning Phase 2 Multi-Proxy Benchmark (20k + 28k)...")
    cmd_train = [
        python_clip, "scratch/train_subcenter_siglip_phase2.py",
        "--mode", "all"
    ]
    subprocess.run(cmd_train, check=True)
    print("\nPhase 2 Pipeline Successfully Completed!")

if __name__ == "__main__":
    main()
