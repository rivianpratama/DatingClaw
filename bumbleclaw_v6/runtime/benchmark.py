"""Latency measurement for a packaged release."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bumbleclaw_v6.runtime.scorer import V6Scorer


def benchmark_release(release_dir: str | Path, image_paths: list[str | Path],
                      old_scores: list[float | None] | None = None) -> dict:
    scorer = V6Scorer(release_dir, controller=False)
    old_scores = old_scores or [None] * len(image_paths)
    timings = []
    proposals = []
    for path, old in zip(image_paths, old_scores):
        started = time.perf_counter()
        result = scorer.decide(path, old)
        timings.append((time.perf_counter() - started) * 1000.0)
        proposals.append({"right": result.get("right"), "reason": result.get("reason"),
                          "p": result.get("p"), "u": result.get("u")})
    return {
        "release": str(release_dir),
        "n": len(timings),
        "p50_ms": float(np.percentile(timings, 50)) if timings else None,
        "p95_ms": float(np.percentile(timings, 95)) if timings else None,
        "max_ms": float(np.max(timings)) if timings else None,
        "proposals": proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark_release(args.release, args.images), indent=2))


if __name__ == "__main__":
    main()
