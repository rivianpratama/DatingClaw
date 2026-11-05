"""Profile-image duplicate scan: exact SHA + pHash (DCT), profile_*.webp only."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dct

def phash(path: Path):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with Image.open(path) as im:
        g = im.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
    c = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    med = np.median(c.flatten()[1:])
    bits = (c.flatten() > med)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return path.name, sha, h

def popcount(x): return int(x).bit_count()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--near", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(p for p in args.folder.glob("profile_*.webp"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(phash, files))
    by_sha = defaultdict(list)
    for name, sha, h in recs: by_sha[sha].append(name)
    exact = [sorted(v) for v in by_sha.values() if len(v) > 1]
    parent = list(range(len(recs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b: parent[b] = a
    pairs = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            d = popcount(recs[i][2] ^ recs[j][2])
            if d <= args.near:
                union(i, j); pairs.append((d, recs[i][0], recs[j][0]))
    groups = defaultdict(list)
    for i in range(len(recs)): groups[find(i)].append(i)
    near = [sorted(g) for g in groups.values() if len(g) > 1]
    meta = {}
    if args.csv and args.csv.exists():
        with args.csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                meta[row.get("screenshot", "")] = {k: row.get(k, "") for k in ("action", "gambler_score", "threshold", "iteration")}
    def describe(idxs):
        names = [recs[i][0] for i in idxs]
        return {"count": len(names), "files": names,
                "actions": [meta.get(n, {}).get("action", "") for n in names],
                "scores": [meta.get(n, {}).get("gambler_score", "") for n in names],
                "iterations": [meta.get(n, {}).get("iteration", "") for n in names]}
    out = {"folder": str(args.folder), "n_images": len(recs), "near_threshold": args.near,
           "exact_duplicate_groups": len(exact), "exact_duplicate_images": sum(len(g) for g in exact),
           "near_duplicate_groups": len(near), "near_duplicate_images": sum(len(g) for g in near),
           "unique_images": len(recs) - sum(len(g) - 1 for g in near),
           "exact_groups": [describe([next(i for i,r in enumerate(recs) if r[0]==n) for n in g]) for g in exact],
           "near_groups": [describe(g) for g in near],
           "closest_pairs": sorted(pairs)[:20]}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if not k.endswith("_groups") and k!="closest_pairs"}, indent=2))
    print("top groups:")
    for g in sorted(out["near_groups"], key=lambda x:-x["count"])[:12]:
        print(g["count"], g["files"][:5], g["actions"][:5], g["scores"][:5])
    print("closest pairs:", out["closest_pairs"][:5])

if __name__ == "__main__": main()
