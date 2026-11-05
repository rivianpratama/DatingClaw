from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def main() -> int:
    root_dir = Path(__file__).resolve().parent
    python_base = root_dir / ".venv" / "Scripts" / "python.exe"
    python_clip = Path(r"D:\BumbleClawClipVenv\Scripts\python.exe")
    hf_cache = r"D:\BumbleClawHFCache"

    labels_path = r"D:\BumbleLog\labels_10k_pure.csv"
    face_store_path = root_dir / "embeddings" / "reference_store_bumble_10k_pure.npz"
    clip_store_path = root_dir / "embeddings" / "clip_store_bumble_10k_pure.npz"
    
    face_reg_path = root_dir / "models" / "rating_regressor_bumble_10k_pure.joblib"
    face_eval_report = root_dir / "results" / "regressor_eval_bumble_10k_pure.csv"
    
    multi_reg_path = root_dir / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib"
    multi_eval_report = root_dir / "results" / "multimodal_regressor_eval_bumble_10k_pure.csv"

    print("================================================================================")
    print("STAGE 1: CLIP Store Feature Extraction (Batch Size 64 on CUDA)")
    print("================================================================================")
    temp_dir = Path(r"D:\BumbleTemp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    torch_cache = Path(r"D:\BumbleClawTorchCache")
    torch_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = hf_cache
    env["TRANSFORMERS_CACHE"] = hf_cache
    env["TORCH_HOME"] = str(torch_cache)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TMPDIR"] = str(temp_dir)

    clip_cmd = [
        str(python_clip),
        str(root_dir / "build_clip_store.py"),
        "--labels", str(labels_path),
        "--store", str(face_store_path),
        "--output", str(clip_store_path),
        "--provider", "cuda",
        "--batch-size", "64",
    ]
    print("Running:", " ".join(clip_cmd))
    res = subprocess.run(clip_cmd, env=env, check=True)
    print(f"CLIP store generated successfully: {clip_store_path}\n")

    print("================================================================================")
    print("STAGE 2: Train InsightFace Rating Regressor")
    print("================================================================================")
    face_reg_cmd = [
        str(python_base),
        str(root_dir / "train_regressor.py"),
        "--store", str(face_store_path),
        "--output", str(face_reg_path),
        "--report", str(face_eval_report),
    ]
    print("Running:", " ".join(face_reg_cmd))
    subprocess.run(face_reg_cmd, env=env, check=True)
    print(f"Face regressor saved to: {face_reg_path}\n")

    print("================================================================================")
    print("STAGE 3: Train Multimodal Rating Regressor (Face + CLIP)")
    print("================================================================================")
    multi_reg_cmd = [
        str(python_base),
        str(root_dir / "train_multimodal_regressor.py"),
        "--face-store", str(face_store_path),
        "--clip-store", str(clip_store_path),
        "--output", str(multi_reg_path),
        "--report", str(multi_eval_report),
    ]
    print("Running:", " ".join(multi_reg_cmd))
    subprocess.run(multi_reg_cmd, env=env, check=True)
    print(f"Multimodal regressor saved to: {multi_reg_path}\n")

    print("================================================================================")
    print("ALL TRAINING STAGES COMPLETED FOR 10K PURE MODEL!")
    print("================================================================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
