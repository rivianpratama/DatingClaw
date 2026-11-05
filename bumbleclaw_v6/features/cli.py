"""Command-line extraction into a recipe-bound feature store."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.features.extract import extract_image, load_models
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.provenance import hash_id_list
from bumbleclaw_v6.schemas import FeatureStatus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recipe", default="photo_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", nargs=2, type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--siglip-model", default=None)
    parser.add_argument("--dinov3-path", default=None)
    args = parser.parse_args()
    C.setup_env()
    recipe = get_recipe(args.recipe)
    if args.siglip_model or args.dinov3_path:
        recipe = type(recipe)(
            recipe.name, recipe.blocks, recipe.views,
            siglip_model=args.siglip_model or recipe.siglip_model,
            dinov3_path=args.dinov3_path or recipe.dinov3_path,
            version=recipe.version,
        )
    snapshot = load_snapshot(args.snapshot)
    samples = snapshot["samples"]
    rows = np.arange(len(samples)) if args.rows is None else np.arange(args.rows[0], args.rows[1])
    if args.limit:
        rows = rows[:args.limit]
    dims = {"arc": 512, "geom": 16, "fv": 1, "sf": 1152, "sp": 1152, "sp_flip": 1152,
            "sp_tight": 1152, "sp_tight_flip": 1152, "sp_tight_tta": 1152,
            "sp_wide": 1152, "stop": 1152, "d3": 1536}
    dims = {block: dims[block] for block in recipe.blocks}
    expected_ids = samples["sample_id"].to_numpy()[rows]
    expected_hash = hash_id_list(expected_ids)
    if args.out.exists():
        store = FeatureStore.open(args.out, recipe)
        if store.manifest["n"] != len(rows) or store.manifest["sample_ids_sha256"] != expected_hash:
            raise ValueError("existing feature store does not match the requested snapshot rows")
        if (args.out / "COMPLETE.json").exists():
            print(f"already complete: {args.out}")
            return
        start = store.next_row
        print(f"resuming {args.out} at row {start}/{len(rows)}", flush=True)
    else:
        store = FeatureStore.create(args.out, expected_ids, recipe, dims)
        start = 0
    models = load_models(
        recipe,
        need_arc="arc" in recipe.blocks,
        need_siglip=any(b in recipe.blocks for b in ("sf", "sp", "sp_flip", "sp_tight", "sp_tight_flip", "sp_tight_tta", "sp_wide", "stop")),
        need_dinov3="d3" in recipe.blocks,
    )
    for i in range(start, len(rows)):
        row = rows[i]
        bundle = extract_image(samples["path"].iloc[row], recipe, models)
        values = {block: bundle.values[block][None] for block in recipe.blocks}
        status = {block: np.array([int(bundle.status[block])], dtype=np.uint8) for block in recipe.blocks}
        store.write_chunk(i, values, status)
        if i % 10 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)} {samples['sample_id'].iloc[row]} {bundle.meta.get('layout')}", flush=True)
    store.finalize()
    print(args.out)


if __name__ == "__main__":
    main()
