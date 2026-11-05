"""Parameter-efficient adaptation of the SigLIP portrait tower.

Training roles for a fold:
  fit  = rows before inner_val
  val  = the 4,540 rows immediately before the test deck
  test = the named fold's test deck

The frozen side features are ArcFace, SigLIP-full, DINOv3 and face validity. The
adapter and the small classifier are trained; the base vision tower is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform
from bumbleclaw_v6.eval.artifacts import RunArtifacts
from bumbleclaw_v6.eval.metrics import deck_metrics
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.features.extract import siglip_model
from bumbleclaw_v6.features.preprocess import crop_portrait
from bumbleclaw_v6.provenance import stable_hash

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIDE_KEYS = ("arc", "sf", "d3", "fv")


def _layer_out(value):
    return value[0] if isinstance(value, tuple) else value


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.scale = alpha / float(r)
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        delta = F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale
        return self.base(x) + delta


class LoRASigLIP(nn.Module):
    def __init__(self, first_layer: int, r: int, alpha: int, dropout: float, d_side: int,
                 targets: tuple[str, ...] = ("q_proj", "v_proj")):
        super().__init__()
        self.vm = siglip_model()
        self.first = int(first_layer)
        for parameter in self.vm.parameters():
            parameter.requires_grad_(False)
        for layer_index in range(self.first, len(self.vm.encoder.layers)):
            layer = self.vm.encoder.layers[layer_index]
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    continue
                for name in targets:
                    module = getattr(parent, name, None)
                    if isinstance(module, nn.Linear):
                        setattr(parent, name, LoRALinear(module, r, alpha, dropout))
        self.vm.head.float()
        for parameter in self.vm.head.parameters():
            parameter.requires_grad_(True)
        self.clf = nn.Linear(1152 + d_side, 1).to(DEV)

    def encode(self, px: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.vm.embeddings(px)
            for layer in self.vm.encoder.layers[:self.first]:
                hidden = _layer_out(layer(hidden, None))
        hidden = hidden.detach()
        for layer in self.vm.encoder.layers[self.first:]:
            if self.training:
                hidden = checkpoint(layer, hidden, None, use_reentrant=False)
                hidden = _layer_out(hidden)
            else:
                hidden = _layer_out(layer(hidden, None))
        pooled = self.vm.head(self.vm.post_layernorm(hidden)).float()
        return F.normalize(pooled, dim=-1)

    def forward(self, px: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        z = self.encode(px) * math.sqrt(1152.0)
        return self.clf(torch.cat([z, side], dim=1)).squeeze(1)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items()
                if ".A" in name or ".B" in name or name.startswith("vm.head") or name.startswith("clf")}


class PortraitDS(Dataset):
    def __init__(self, paths: list[str], bboxes: np.ndarray, rows: np.ndarray, train: bool):
        self.paths = paths
        self.bboxes = bboxes
        self.rows = np.asarray(rows, dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        image = cv2.imread(self.paths[row])
        if image is None:
            x = np.zeros((3, 384, 384), np.float32)
        else:
            scale = float(np.random.uniform(1.6, 2.0)) if self.train else 1.8
            crop, _ = crop_portrait(image, self.bboxes[row], 384, scale)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)
            if self.train and np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x), row


def load_bboxes() -> np.ndarray:
    with np.load(C.TRAIN_D3_NPZ, allow_pickle=True) as train:
        train_boxes = train["bboxes"].astype(np.float32)
    with np.load(C.HOLDOUT_NPZ, allow_pickle=True) as holdout:
        holdout_boxes = holdout["bboxes"].astype(np.float32)
    return np.vstack([train_boxes, holdout_boxes])


@torch.no_grad()
def score_rows(model, ds_args, side_t, rows, batch_size=32, workers=0) -> np.ndarray:
    model.eval()
    loader = DataLoader(PortraitDS(*ds_args, rows=rows, train=False), batch_size=batch_size,
                        num_workers=workers, shuffle=False)
    out = []
    for px, row in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(px.to(DEV).half(), side_t[row.to(DEV)])
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fold", default="F5")
    parser.add_argument("--run", default="e05r_lora_sp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--neg-frac", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--first-layer", type=int, default=23)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side", choices=["legacy", "photo"], default="legacy")
    parser.add_argument("--feature-store", type=Path, default=Path("D:/bumbleclaw_v6r1/features/photo_v1_full_r2"))
    parser.add_argument("--full", action="store_true", help="Train on all labels for fixed epochs; no val/test selection")
    parser.add_argument("--boxes", type=Path, default=None, help="Runtime-detector box npz (features.boxes); replaces legacy boxes")
    parser.add_argument("--cv-fold", type=int, default=-1, help="With --full: hold out this grouped fold and save its OOF logits")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    args = parser.parse_args()
    C.setup_env()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    snapshot = load_snapshot(args.snapshot)
    meta = snapshot["samples"]
    y = meta["y"].to_numpy(dtype=np.int64)
    ids = meta["sample_id"].astype(str).to_numpy()
    old = meta["old_score"].to_numpy(dtype=np.float64)
    oof_rows = np.empty(0, dtype=np.int64)
    if args.full and args.cv_fold >= 0:
        from bumbleclaw_v6.experiments.full_release import _groups, grouped_folds
        fold_ids = grouped_folds(_groups(meta), n_folds=args.cv_folds, seed=args.cv_seed)
        fit_rows = np.flatnonzero(fold_ids != args.cv_fold).astype(np.int64)
        oof_rows = np.flatnonzero(fold_ids == args.cv_fold).astype(np.int64)
        val_rows = test_rows = np.empty(0, dtype=np.int64)
    elif args.full:
        fit_rows = np.arange(len(meta), dtype=np.int64)
        val_rows = test_rows = np.empty(0, dtype=np.int64)
    else:
        make_matched_roles(meta, args.fold)
        s = int(C.FOLD_STARTS[int(args.fold[1:]) - 1])
        fit_rows = np.arange(0, s - C.FOLD_LEN, dtype=np.int64)
        val_rows = np.arange(s - C.FOLD_LEN, s, dtype=np.int64)
        test_rows = np.arange(s, s + C.FOLD_LEN, dtype=np.int64)
    if args.side == "photo":
        from bumbleclaw_v6.features.bank import FeatureStoreBank
        from bumbleclaw_v6.features.recipes import get_recipe
        side_keys = ("arc", "d3", "fv")
        blocks = FeatureStoreBank(args.snapshot, args.feature_store, get_recipe("photo_v1")).blocks(side_keys)
    else:
        side_keys = SIDE_KEYS
        blocks = LegacyFeatureBank(args.snapshot, SIDE_KEYS).blocks(SIDE_KEYS)
    specs = [BlockSpec(key, "unit", True, True) for key in side_keys if key != "fv"]
    specs.append(BlockSpec("fv", "passthrough", False, False))
    transform = FeatureTransform.fit(blocks, fit_rows, specs)
    side, _ = transform.transform(blocks)
    if args.boxes is not None:
        with np.load(args.boxes, allow_pickle=False) as zb:
            if not np.array_equal(zb["sample_id"].astype(str), ids):
                raise ValueError("box file sample_id order does not match snapshot")
            if not zb["done"].all():
                raise ValueError("box file is incomplete")
            bboxes = zb["bbox"].astype(np.float32)
    else:
        bboxes = load_bboxes()
    paths = meta["path"].astype(str).tolist()
    ds_args = (paths, bboxes)
    side_t = torch.tensor(side, device=DEV, dtype=torch.float32)
    model = LoRASigLIP(args.first_layer, args.r, args.alpha, args.dropout, side.shape[1], tuple(args.targets))
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vm.head")]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-4},
        {"params": model.clf.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    positives = fit_rows[y[fit_rows] == 1]
    negatives = fit_rows[y[fit_rows] == 0]
    epoch_size = len(positives) + int(args.neg_frac * len(negatives))
    steps_per_epoch = math.ceil(epoch_size / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(200, max(1, steps_per_epoch // 5))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    val_eval = val_rows[:args.eval_limit] if args.eval_limit else val_rows
    test_eval = test_rows[:args.eval_limit] if args.eval_limit else test_rows
    run = RunArtifacts.create(args.run, C.RESULTS_DIR, {
        "kind": "lora_siglip_portrait",
        "snapshot": str(args.snapshot),
        "fold": args.fold,
        "fit_rows": int(len(fit_rows)),
        "val_rows": int(len(val_rows)),
        "test_rows": int(len(test_rows)),
        "first_layer": args.first_layer,
        "r": args.r,
        "alpha": args.alpha,
        "targets": args.targets,
        "side_keys": list(side_keys),
        "side": args.side,
        "full_fit": bool(args.full),
        "boxes": str(args.boxes) if args.boxes is not None else "legacy",
        "cv_fold": int(args.cv_fold), "cv_folds": int(args.cv_folds), "cv_seed": int(args.cv_seed),
        "feature_store": str(args.feature_store) if args.side == "photo" else None,
        "transform_hash": stable_hash(transform.to_dict()),
    })
    best = (-1, None, 0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sampled_neg = np.random.choice(negatives, epoch_size - len(positives), replace=False)
        rows = np.random.permutation(np.concatenate([positives, sampled_neg]))
        loader = DataLoader(PortraitDS(paths, bboxes, rows, train=True), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers, drop_last=True)
        started = time.time()
        for i, (px, row) in enumerate(loader):
            if args.max_steps and i >= args.max_steps:
                break
            progress = min(1.0, (step + 1) / max(1, warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, total_steps))))
            factor = progress * cosine
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            with torch.autocast("cuda", dtype=torch.float16):
                row_gpu = row.to(DEV)
                logits = model(px.to(DEV).half(), side_t[row_gpu])
                loss = F.binary_cross_entropy_with_logits(logits, torch.tensor(y[row], device=DEV, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if i % 50 == 0:
                print(f"ep{epoch} step {i}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"{(i + 1) * args.batch / max(1e-6, time.time() - started):.1f} img/s "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        if args.full:
            torch.save(model.trainable_state(), run.root / "models" / f"full_ep{epoch}_lora.pt")
            print(f"[ep{epoch}] saved ({time.time() - started:.1f}s)", flush=True)
            continue
        val_logits = score_rows(model, ds_args, side_t, val_eval, batch_size=32, workers=0)
        val_metrics = deck_metrics(val_logits, y[val_eval], ids[val_eval], old_score=old[val_eval])
        print(f"[ep{epoch}] val hits {val_metrics['hits']}/{val_metrics['n_pos']} "
              f"severe {val_metrics['proxy_severe_fp']} ({time.time() - started:.1f}s)", flush=True)
        if val_metrics["hits"] > best[0]:
            best = (val_metrics["hits"], model.trainable_state(), epoch)
    if args.full:
        torch.save(model.trainable_state(), run.root / "models" / "full_final_lora.pt")
        torch.save(model.clf.state_dict(), run.root / "models" / "full_final_clf.pt")
        (run.root / "transform.json").write_text(json.dumps(transform.to_dict()), encoding="utf-8")
        if len(oof_rows) and args.eval_limit:
            oof_rows = oof_rows[:args.eval_limit]  # smoke tests only
        if len(oof_rows):
            oof_logits = score_rows(model, ds_args, side_t, oof_rows, batch_size=32, workers=0)
            if not np.isfinite(oof_logits).all():
                raise FloatingPointError("OOF logits contain non-finite values")
            np.savez(run.root / "oof_logits.npz", sample_id=ids[oof_rows], row=oof_rows, logit=oof_logits)
        summary = {"run_id": args.run, "full_fit": True, "epochs": args.epochs, "fit_rows": int(len(fit_rows)),
                   "oof_rows": int(len(oof_rows)), "boxes": str(args.boxes) if args.boxes is not None else "legacy",
                   "side_keys": list(side_keys), "evaluation": "none; R22 prospective only",
                   "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        run.finalize(summary)
        print(json.dumps(summary, indent=2))
        return
    if best[1] is None:
        raise RuntimeError("training produced no finite validation result")
    model.load_state_dict(best[1], strict=False)
    val_logits = score_rows(model, ds_args, side_t, val_eval, batch_size=32, workers=0)
    test_logits = score_rows(model, ds_args, side_t, test_eval, batch_size=32, workers=0)
    run.save_predictions(args.fold, "H", args.seed, ids[val_eval], val_logits)
    run.save_predictions(args.fold, "T", args.seed, ids[test_eval], test_logits)
    torch.save(best[1], run.root / "models" / f"{args.fold}_s{args.seed}_lora.pt")
    test_metrics = deck_metrics(test_logits, y[test_eval], ids[test_eval], old_score=old[test_eval])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    test_metrics = {key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
                    for key, value in test_metrics.items()}
    summary = {"run_id": args.run, "fold": args.fold, "best_epoch": best[2],
               "val_hits": best[0], "test": test_metrics, "config": config}
    run.finalize(summary)
    print(json.dumps({"run": str(run.root), "best_epoch": best[2], "val_hits": best[0],
                      "test_hits": test_metrics["hits"], "severe": test_metrics["proxy_severe_fp"]}, indent=2))


if __name__ == "__main__":
    main()
