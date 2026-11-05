"""Scan a folder of profile images for exact and near-duplicate images.

Exact: SHA-256. Near: 64-bit dHash + aHash, Hamming distance <= threshold.
Also joins scores.csv so duplicate groups show actions/scores.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

def hashes(path: Path):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with Image.open(path) as im:
        g = im.convert("L")
        d = np.asarray(g.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        dbits = (d[:, 1:] > d[:, :-1]).flatten()
        dh = 0
        for b in dbits:
            dh = (dh << 1) | int(b)
        a = np.asarray(g.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
        ah = 0
        for b in (a > a.mean()).flatten():
            ah = (ah << 1) | int(b)
    return path.name, sha, dh, ah, im.size

def popcount(x: int) -> int:
    return int(x).bit_count()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--near", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = sorted([p for p in args.folder.iterdir() if p.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}])
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(hashes, files))
    by_sha = defaultdict(list)
    for name, sha, dh, ah, size in recs:
        by_sha[sha].append(name)
    exact = [sorted(v) for v in by_sha.values() if len(v) > 1]
    # union-find over near-duplicate pairs
    parent = list(range(len(recs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b: parent[b] = a
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if popcount(recs[i][2] ^ recs[j][2]) <= args.near or popcount(recs[i][3] ^ recs[j][3]) <= args.near:
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    near = [sorted(g) for g in groups.values() if len(g) > 1]
    # join CSV
    meta = {}
    if args.csv and args.csv.exists():
        with args.csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                meta[row.get("screenshot", "")] = {k: row.get(k, "") for k in ("action", "gambler_score", "threshold", "iteration")}
    def describe(idxs):
        names = [recs[i][0] for i in idxs]
        return {
            "count": len(names),
            "files": names,
            "actions": [meta.get(n, {}).get("action", "") for n in names],
            "scores": [meta.get(n, {}).get("gambler_score", "") for n in names],
            "iterations": [meta.get(n, {}).get("iteration", "") for n in names],
        }
    exact_desc = [describe([next(i for i, r in enumerate(recs) if r[0] == n) for n in g]) for g in exact]
    near_desc = [describe(g) for g in near]
    out = {
        "folder": str(args.folder), "n_images": len(recs), "near_threshold": args.near,
        "exact_duplicate_groups": len(exact), "exact_duplicate_images": sum(len(g) for g in exact),
        "near_duplicate_groups": len(near), "near_duplicate_images": sum(len(g) for g in near),
        "unique_images": len(recs) - sum(len(g) - 1 for g in near),
        "exact_groups": exact_desc, "near_groups": near_desc,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_groups")}, indent=2))
    print("top near groups:")
    for g in sorted(near_desc, key=lambda x: -x["count"])[:10]:
        print(g["count"], g["files"][:4], g["actions"][:4], g["scores"][:4])

if __name__ == "__main__":
    main()
