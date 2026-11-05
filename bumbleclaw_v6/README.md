# BumbleClaw v6 revision package

This package implements the safe foundation of `planning/v6_revision/`: immutable data
snapshots, explicit fit/select/calibrate/policy/test roles, strict metrics, model-bound
calibration, expected-cost policy, streaming controllers, and reproducible artifacts.

It deliberately does **not** include production image extraction, a live scorer, phone
integration, human-study UI, LoRA, or live activation. Those require the approvals in the
task board.

## Environment

Run commands from the repository root:

```powershell
$py = "D:\BumbleClawClipVenv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
```

Artifacts go to `D:\bumbleclaw_v6r1`. Override with `BUMBLECLAW_V6R1_DIR` only when you
intend a separate experiment tree. Never point it at the repository or a source store.

## Tests

```powershell
& $py -m pytest bumbleclaw_v6/tests -q
```

## Snapshot

```powershell
& $py -m bumbleclaw_v6.data.build_snapshot
```

The canonical snapshot currently used by this checkout:

```text
D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935
```

The snapshot preserves all label events and flags four conflicting filenames. Historical
samples use `first_event_v1`; legacy feature arrays are referenced, not copied. Runs are
currently `filename_only` group isolation, not identity-clean.

## Baseline smoke

```powershell
& $py -m bumbleclaw_v6.experiments.baselines `
  --snapshot D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935 `
  --configs arc_portrait --folds F5 --seeds 0 --wd-grid 1.0 --run-prefix my_run
```

Each fold uses A fit → H selection → A∪H refit → C calibration role → P policy role →
T test. No C/P/T label is used to fit or select the model.

## Policy smoke

```powershell
& $py -m bumbleclaw_v6.experiments.policy_sweep `
  --base-run D:\bumbleclaw_v6r1\results\my_run_arc_portrait `
  --snapshot D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935 `
  --budgets 10 --folds F5 --run-prefix my_policy
```

The budget is per 4,540 candidates and is rescaled to the calibration/policy role size.
`ABSTAIN` is the default unknown-cost behavior; missing cost is never silently safe.

## Comparison

```powershell
& $py -m bumbleclaw_v6.eval.compare `
  --a D:\bumbleclaw_v6r1\results\run_a `
  --b D:\bumbleclaw_v6r1\results\run_b `
  --snapshot D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935 `
  --folds F5 --n-boot 200
```

`PROMOTE_DIAGNOSTIC` is only a development heuristic. It is not release approval and
does not correct for repeated candidate selection.

## Current evidence boundary

The checked-in development results are **F2–F5, one seed, one weight decay, legacy
features, filename-only groups**. They demonstrate that the contracts and artifact flow
work; they are not a full E01/E02 grid and do not establish >90% recall or production
latency.
