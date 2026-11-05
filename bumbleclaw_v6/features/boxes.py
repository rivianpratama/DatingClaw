"""Recompute face boxes with the exact runtime extraction path (resumable).

Output npz: sample_id (snapshot order), bbox (N,4; -1 when no face or the image is
unsupported), face_valid (N,), layout (N,), done (N,). Uses ``extract_image`` with
blocks (arc, fv), so detection is byte-for-byte the runtime detector.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import load_snapshot


def _save(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (2 ** attempt))
    raise PermissionError(f"could not replace {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    args = parser.parse_args()
    from bumbleclaw_v6.features.extract import extract_image, load_models
    from bumbleclaw_v6.features.preprocess import PHOTO_V1
    from bumbleclaw_v6.features.recipes import FeatureRecipe

    meta = load_snapshot(args.snapshot)["samples"]
    ids = np.asarray(meta["sample_id"].astype(str).tolist(), dtype="<U64")
    paths = meta["path"].astype(str).tolist()
    n = len(ids) if not args.limit else min(args.limit, len(ids))
    if args.out.exists():
        with np.load(args.out, allow_pickle=False) as z:  # context manager: release the Windows file lock
            if not np.array_equal(z["sample_id"].astype(str), ids):
                raise ValueError("existing box file sample_id order differs from snapshot")
            bbox, fv, layout, done = z["bbox"].copy(), z["face_valid"].copy(), z["layout"].astype("<U20").copy(), z["done"].copy()
    else:
        bbox = np.full((len(ids), 4), -1.0, np.float32)
        fv = np.zeros(len(ids), np.int8)
        layout = np.full(len(ids), "", dtype="<U20")
        done = np.zeros(len(ids), bool)
    recipe = FeatureRecipe("boxes_runtime_v1", ("arc", "fv"), (PHOTO_V1,))
    models = load_models(recipe, need_arc=True, need_siglip=False, need_dinov3=False)
    started, count = time.time(), 0
    for i in range(n):
        if done[i]:
            continue
        bundle = extract_image(paths[i], recipe, models)
        b = bundle.meta.get("bbox")
        bbox[i] = np.asarray(b, np.float32) if b is not None else -1.0
        fv[i] = int(bundle.meta.get("face_valid", 0))
        layout[i] = str(bundle.meta.get("layout", ""))
        done[i] = True
        count += 1
        if count % args.checkpoint_every == 0:
            _save(args.out, sample_id=ids, bbox=bbox, face_valid=fv, layout=layout, done=done)
            rate = count / (time.time() - started)
            print(f"{int(done.sum())}/{len(ids)} {rate:.1f} img/s eta {(n - done[:n].sum()) / rate / 60:.1f} min", flush=True)
    _save(args.out, sample_id=ids, bbox=bbox, face_valid=fv, layout=layout, done=done)
    print(json.dumps({"done": int(done.sum()), "n": len(ids), "face_valid": int(fv.sum())}))


if __name__ == "__main__":
    main()
