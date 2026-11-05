"""Wait for crop_v1_full, then run E05 tight/wide baselines + policy and report."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
PY = Path(r"D:\BumbleClawClipVenv\Scripts\python.exe")
SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full")
RESULTS = Path(r"D:\bumbleclaw_v6r1\results")
REPORT = RESULTS / "E05_REPORT.md"
LOG = RESULTS / "auto_after_e05.log"
DECISIONS = ROOT / "planning" / "v6_revision" / "12_DECISIONS_AND_RESULTS.md"
RUN_PREFIX = "e15r_crop"
POLICY_PREFIX = "e15p_crop"
CONFIGS = ("no_sf_tight", "no_sf_wide")
MAX_WAIT_S = 6 * 3600


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def process_running() -> bool:
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
           "| Select-Object -ExpandProperty CommandLine")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    return "crop_v1_full" in r.stdout


def run(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


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
        REPORT.write_text("# E05 report\n\nExtraction did not complete; no comparison run.\n", encoding="utf-8")
        log("extraction incomplete; stopping")
        return
    log("extraction complete; running baselines")
    env = {"PYTHONPATH": ".", "TEMP": r"D:\BumbleTemp"}
    cmd = [str(PY), "-m", "bumbleclaw_v6.experiments.baselines",
           "--snapshot", str(SNAP), "--feature-store", str(STORE), "--feature-recipe", "crop_v1",
           "--configs", *CONFIGS, "--folds", "F2", "F3", "F4", "F5",
           "--seeds", "0", "1", "2", "--wd-grid", "0.1", "0.3", "1.0",
           "--run-prefix", RUN_PREFIX]
    rc = run(cmd, RESULTS / "e15r_crop_baselines.log")
    log(f"baselines rc={rc}")
    if rc != 0:
        REPORT.write_text("# E05 report\n\nBaseline run failed; see e15r_crop_baselines.log.\n", encoding="utf-8")
        return
    for cfg in CONFIGS:
        cmd = [str(PY), "-m", "bumbleclaw_v6.experiments.policy_sweep",
               "--base-run", str(RESULTS / f"{RUN_PREFIX}_{cfg}"), "--snapshot", str(SNAP),
               "--budgets", "10", "--folds", "F2", "F3", "F4", "F5",
               "--run-prefix", f"{POLICY_PREFIX}_{cfg}"]
        rc = run(cmd, RESULTS / f"e15p_crop_{cfg}.log")
        log(f"policy {cfg} rc={rc}")
    # Summarize.
    def policy_summary(run_id: str) -> dict:
        d = json.loads((RESULTS / run_id / "summary.json").read_text(encoding="utf-8"))
        out = {}
        for fold, fd in d["folds"].items():
            t = fd.get("test")
            if t and t.get("feasible"):
                m = t["metrics"]
                out[fold] = {"hits": m["hits"], "n_pos": m["n_pos"], "severe": m["proxy_severe_fp"],
                             "lambda": fd["policy"]["lambda"]}
        return out

    r14 = json.loads((RESULTS / "e14r_photo_frontier.json").read_text(encoding="utf-8"))
    lines = ["# E05 — photo-view diagnosis", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    lines += ["Comparator: R14 `no_sf` (scale 1.8).", ""]
    lines += ["| Variant | F5 policy hits/severe | Pooled hits/severe |", "|---|---:|---:|"]
    summaries = {}
    for cfg in CONFIGS:
        s = policy_summary(f"{POLICY_PREFIX}_{cfg}_b10")
        summaries[cfg] = s
        f5 = s.get("F5", {})
        pooled_hits = sum(v["hits"] for v in s.values())
        pooled_sev = sum(v["severe"] for v in s.values())
        lines.append(f"| `{cfg}` | {f5.get('hits')}/{f5.get('severe')} | {pooled_hits}/{pooled_sev} |")
    base = r14["policy"]["no_sf"]
    base_f5 = base.get("F5", {})
    base_pool_hits = sum(v.get("hits", 0) for v in base.values())
    base_pool_sev = sum(v.get("severe", 0) for v in base.values())
    lines += ["", f"R14 `no_sf`: F5 {base_f5.get('hits')}/{base_f5.get('severe')}; pooled {base_pool_hits}/{base_pool_sev}.", ""]
    verdicts = []
    for cfg, s in summaries.items():
        f5 = s.get("F5", {})
        pooled_hits = sum(v["hits"] for v in s.values())
        pooled_sev = sum(v["severe"] for v in s.values())
        ok = (f5.get("hits", 0) > base_f5.get("hits", 0) and f5.get("severe", 999) <= base_f5.get("severe", 999)) or \
             (pooled_hits >= base_pool_hits + 4 and pooled_sev <= base_pool_sev)
        verdicts.append((cfg, ok))
    lines += ["## Verdict", ""]
    for cfg, ok in verdicts:
        lines.append(f"- `{cfg}`: {'PROMISING' if ok else 'REJECTED'} by the predeclared stop rule.")
    lines += ["", "No F5 threshold tuning; selection remains on H/P.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write("\n\n### R15 E05 result — " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        f.write(REPORT.read_text(encoding="utf-8"))
    log("report written")


if __name__ == "__main__":
    main()
