"""Detached v6 post-LoRA watcher.

Waits for e05r_lora_f5_long_r2 to finish, compares it to the base and short
LoRA runs, appends the verdict to the v6 results log, and starts R13 extraction
(photo_v1_full_r2) only after the LoRA process is gone. CPU-only while waiting.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
PY = Path(r"D:\BumbleClawClipVenv\Scripts\python.exe")
SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
RUN = Path(r"D:\bumbleclaw_v6r1\results\e05r_lora_f5_long_r2")
BASE = Path(r"D:\bumbleclaw_v6r1\results\e01r_dev4_base")
SHORT = Path(r"D:\bumbleclaw_v6r1\results\e05r_lora_f5")
OUTDIR = Path(r"D:\bumbleclaw_v6r1\results")
REPORT = OUTDIR / "AUTO_AFTER_LORA_REPORT.md"
WATCH_LOG = OUTDIR / "auto_after_lora.log"
RESULTS = ROOT / "planning" / "v6_revision" / "12_DECISIONS_AND_RESULTS.md"
R13_OUT = Path(r"D:\bumbleclaw_v6r1\features\photo_v1_full_r2")
R13_LOG = OUTDIR / "r13_extract_full_r2.log"
RUN_ID = "e05r_lora_f5_long_r2"
MAX_WAIT_S = 9 * 3600


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    with WATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def process_running(run_id: str) -> bool:
    # Avoid nested PowerShell quoting: fetch command lines and match in Python.
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object -ExpandProperty CommandLine"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True, text=True, timeout=60,
    )
    return run_id in result.stdout


def compare(a: Path, b: Path, out: Path) -> dict | None:
    command = [
        str(PY), "-m", "bumbleclaw_v6.eval.compare",
        "--a", str(a), "--b", str(b), "--snapshot", str(SNAP),
        "--folds", "F5", "--n-boot", "2000",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        log(f"compare failed for {a.name} vs {b.name}: {result.stderr[-500:]}")
        return None
    data = json.loads(result.stdout)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def start_r13() -> tuple[int, Path] | None:
    if R13_OUT.exists():
        log(f"R13 output already exists; not starting: {R13_OUT}")
        return None
    R13_OUT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), "-m", "bumbleclaw_v6.features.cli",
        "--snapshot", str(SNAP), "--recipe", "photo_v1", "--out", str(R13_OUT),
    ]
    log_handle = R13_LOG.open("w", encoding="utf-8")
    # Detached process group so it survives this watcher.
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log(f"R13 started pid={proc.pid} out={R13_OUT} log={R13_LOG}")
    return proc.pid, R13_LOG


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log(f"watcher started for {RUN_ID}")
    started = time.time()
    dead_checks = 0
    complete = False
    while time.time() - started < MAX_WAIT_S:
        if (RUN / "COMPLETE.json").exists():
            complete = True
            break
        try:
            running = process_running(RUN_ID)
        except Exception as exc:
            log(f"process check failed: {exc}")
            running = True
        if running:
            dead_checks = 0
        else:
            dead_checks += 1
            log(f"run process not found ({dead_checks}/3)")
            if dead_checks >= 3:
                break
        time.sleep(60)

    summary = None
    cmp_base = None
    cmp_short = None
    if complete and (RUN / "summary.json").exists():
        summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
        cmp_base = compare(RUN, BASE, OUTDIR / f"{RUN_ID}_compare_vs_base.json")
        cmp_short = compare(RUN, SHORT, OUTDIR / f"{RUN_ID}_compare_vs_short.json")
        status = "COMPLETE"
    else:
        status = "FAILED_OR_TIMEOUT"

    lines = [
        "# Auto post-LoRA report",
        "",
        f"- Run: `{RUN_ID}`",
        f"- Status: **{status}**",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if summary:
        test = summary.get("test", {})
        lines += [
            "",
            "## LoRA F5 test",
            f"- best epoch: {summary.get('best_epoch')}",
            f"- hits: {test.get('hits')} / {test.get('n_pos')}",
            f"- recall: {test.get('recall')}",
            f"- proxy severe FPs: {test.get('proxy_severe_fp')}",
            f"- AUC: {test.get('auc')}",
            f"- precision: {test.get('precision')}",
        ]
    if cmp_base:
        paired = cmp_base.get("paired", {})
        lines += [
            "",
            "## Paired comparison vs base",
            f"- delta hits: {paired.get('delta_hits')}",
            f"- 95% CI: {paired.get('ci95')}",
            f"- base pooled: {cmp_base.get('pooled_b')}",
            f"- LoRA pooled: {cmp_base.get('pooled_a')}",
        ]
    if cmp_short:
        paired = cmp_short.get("paired", {})
        lines += [
            "",
            "## Paired comparison vs 3-epoch LoRA",
            f"- delta hits: {paired.get('delta_hits')}",
            f"- 95% CI: {paired.get('ci95')}",
        ]
    # Conservative single-fold verdict; this is not a release promotion.
    verdict = "INCONCLUSIVE"
    if summary and cmp_base:
        test = summary.get("test", {})
        paired = cmp_base.get("paired", {})
        base_severe = (cmp_base.get("per_fold_b", {}).get("F5", {}) or {}).get("proxy_severe_fp")
        if (paired.get("delta_hits", 0) >= 4 and paired.get("ci95", [0])[0] > 0
                and base_severe is not None
                and test.get("proxy_severe_fp", 10**9) <= base_severe):
            verdict = "CANDIDATE (single-fold diagnostic only; policy/release gate not passed)"
        else:
            verdict = "REJECTED for release (no safe-frontier improvement)"
    lines += ["", f"## Verdict\n\n**{verdict}**", ""]

    # Start R13 only after LoRA process is truly gone.
    r13 = None
    try:
        still_running = process_running(RUN_ID)
    except Exception:
        still_running = True
    if not still_running:
        r13 = start_r13()
    else:
        log("not starting R13: LoRA process still present")
    if r13:
        lines += [
            "## R13 handoff",
            f"- PID: {r13[0]}",
            f"- Output: `{R13_OUT}`",
            f"- Log: `{r13[1]}`",
            "",
        ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### Auto post-LoRA check — " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        handle.write(f"- Run: `{RUN_ID}`\n- Status: **{status}**\n")
        if summary:
            test = summary.get("test", {})
            handle.write(
                f"- F5: hits {test.get('hits')}/{test.get('n_pos')}, "
                f"recall {test.get('recall')}, severe {test.get('proxy_severe_fp')}, "
                f"AUC {test.get('auc')}, best_epoch {summary.get('best_epoch')}\n"
            )
        if cmp_base:
            handle.write(f"- vs base: {cmp_base.get('paired')}\n")
        if cmp_short:
            handle.write(f"- vs short LoRA: {cmp_short.get('paired')}\n")
        handle.write(f"- Verdict: **{verdict}**\n")
        if r13:
            handle.write(f"- R13 started: pid {r13[0]}, log `{r13[1]}`\n")
        else:
            handle.write("- R13 not started (LoRA still running, output exists, or timeout)\n")
    log(f"report written: {REPORT}")


if __name__ == "__main__":
    main()
