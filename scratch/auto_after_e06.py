"""Wait for photo_tta_tight_full, verify, run E06 TTA baseline/policy, report."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.provenance import hash_id_list
from bumbleclaw_v6.schemas import FeatureStatus

ROOT = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
PY = Path(r"D:\BumbleClawClipVenv\Scripts\python.exe")
SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\photo_tta_tight_full")
RESULTS = Path(r"D:\bumbleclaw_v6r1\results")
REPORT = RESULTS / "E06_REPORT.md"
LOG = RESULTS / "auto_after_e06.log"
DECISIONS = ROOT / "planning" / "v6_revision" / "12_DECISIONS_AND_RESULTS.md"
RUN_PREFIX = "e16r_tta"
POLICY_PREFIX = "e16p_tta"
CONFIG = "no_sf_tight_tta"
MAX_WAIT_S = 8 * 3600


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def process_running() -> bool:
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
           "| Select-Object -ExpandProperty CommandLine")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    return "photo_tta_tight_full" in r.stdout


def run(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def verify_store() -> dict:
    recipe = get_recipe("photo_tta_tight_v1")
    store = FeatureStore.load(STORE, recipe)
    ids = load_snapshot(SNAP)["samples"]["sample_id"].astype(str).to_numpy()
    out = {"n": store.manifest["n"], "complete": store.manifest["complete"],
           "sample_ids_match": hash_id_list(ids) == store.manifest["sample_ids_sha256"], "blocks": {}}
    for block in store.manifest["dims"]:
        status = np.asarray(store.status[block])
        values = np.asarray(store.values[block])
        ok = status == int(FeatureStatus.OK)
        out["blocks"][block] = {
            "status": {FeatureStatus(int(v)).name: int((status == v).sum()) for v in np.unique(status)},
            "finite_ok": bool(np.isfinite(values[ok]).all()) if ok.any() else True,
            "zero_non_ok": bool((values[~ok] == 0).all()) if (~ok).any() else True,
        }
    return out


def main() -> None:
    log("watcher started")
    start = time.time()
    dead = 0
    while time.time() - start < MAX_WAIT_S:
        if (STORE / "COMPLETE.json").exists():
            break
        if process_running():
            dead = 0
        else:
            dead += 1
            log(f"extraction process not found ({dead}/3)")
            if dead >= 3:
                break
        time.sleep(60)
    if not (STORE / "COMPLETE.json").exists():
        REPORT.write_text("# E06 report\n\nExtraction did not complete; no comparison run.\n", encoding="utf-8")
        log("extraction incomplete; stopping")
        return
    verification = verify_store()
    (RESULTS / "e06_tta_store_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    log("extraction complete and verified; running baselines")
    env = {"PYTHONPATH": ".", "TEMP": r"D:\BumbleTemp"}
    cmd = [str(PY), "-m", "bumbleclaw_v6.experiments.baselines",
           "--snapshot", str(SNAP), "--feature-store", str(STORE), "--feature-recipe", "photo_tta_tight_v1",
           "--configs", CONFIG, "--folds", "F2", "F3", "F4", "F5",
           "--seeds", "0", "1", "2", "--wd-grid", "0.1", "0.3", "1.0",
           "--run-prefix", RUN_PREFIX]
    rc = run(cmd, RESULTS / "e16r_tta_baselines.log")
    log(f"baselines rc={rc}")
    if rc != 0:
        REPORT.write_text("# E06 report\n\nBaseline run failed; see e16r_tta_baselines.log.\n", encoding="utf-8")
        return
    cmd = [str(PY), "-m", "bumbleclaw_v6.experiments.policy_sweep",
           "--base-run", str(RESULTS / f"{RUN_PREFIX}_{CONFIG}"), "--snapshot", str(SNAP),
           "--budgets", "10", "--folds", "F2", "F3", "F4", "F5",
           "--run-prefix", f"{POLICY_PREFIX}_{CONFIG}"]
    rc = run(cmd, RESULTS / f"e16p_tta_{CONFIG}.log")
    log(f"policy rc={rc}")
    if rc != 0:
        REPORT.write_text("# E06 report\n\nPolicy run failed; see e16p_tta log.\n", encoding="utf-8")
        return
    # Compare against E05 tight policy and R14 no_sf.
    for label, other in (("tight", "e15p_crop_no_sf_tight_b10"), ("base", "e14p_photo_no_sf_b10")):
        cmd = [str(PY), "-m", "bumbleclaw_v6.eval.compare",
               "--a", str(RESULTS / f"{POLICY_PREFIX}_{CONFIG}_b10"), "--b", str(RESULTS / other),
               "--snapshot", str(SNAP), "--folds", "F2", "F3", "F4", "F5", "--n-boot", "2000"]
        with (RESULTS / f"e06_compare_vs_{label}.json").open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    summary = json.loads((RESULTS / f"{POLICY_PREFIX}_{CONFIG}_b10" / "summary.json").read_text(encoding="utf-8"))
    pol = {}
    for fold, fd in summary["folds"].items():
        t = fd.get("test")
        if t and t.get("feasible"):
            m = t["metrics"]
            pol[fold] = {"hits": m["hits"], "n_pos": m["n_pos"], "severe": m["proxy_severe_fp"],
                         "lambda": fd["policy"]["lambda"]}
    f5 = pol.get("F5", {})
    pooled_hits = sum(v["hits"] for v in pol.values())
    pooled_sev = sum(v["severe"] for v in pol.values())
    cmp_tight = json.loads((RESULTS / "e06_compare_vs_tight.json").read_text(encoding="utf-8"))
    lines = ["# E06 — flip/TTA on tight crop", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
             "Comparator: E05 `no_sf_tight` (F5 155/0; pooled 703/11).", "",
             "| Variant | F5 policy hits/severe | Pooled hits/severe |", "|---|---:|---:|",
             f"| `no_sf_tight_tta` | {f5.get('hits')}/{f5.get('severe')} | {pooled_hits}/{pooled_sev} |", "",
             f"Paired vs E05 tight: delta hits {cmp_tight['paired']['delta_hits']}, "
             f"CI {cmp_tight['paired']['ci95']}.", ""]
    ok = (f5.get("hits", 0) > 155 and f5.get("severe", 999) <= 0) or (pooled_hits >= 707 and pooled_sev <= 11)
    lines += ["## Verdict", "", f"**{'PROMISING' if ok else 'REJECTED'}** by the predeclared stop rule.", "",
              "No F5 threshold tuning.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write("\n\n### R15 E06 result — " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        f.write(REPORT.read_text(encoding="utf-8"))
    log("report written")


if __name__ == "__main__":
    main()
