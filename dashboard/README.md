# DatingClaw Dashboard

This Next.js app is a local inspection surface for dating-app automation logs. It reads score history, setup metadata, screenshots, threshold state, and gallery data from the local automation workflow so model behavior can be checked after a run.

The dashboard is intended for private local data. Score logs and screenshots can reveal personal information; review the log directory and network boundary before exposing the app outside the machine that owns those files.

## Run

Start the Python analysis API from the repo root before using `/sandbox`:

```powershell
.\.venv\Scripts\python.exe app.py
```

Use the configured CLIP venv instead when `/sandbox` should compute CLIP-backed multimodal scores.

Then start the dashboard:

```powershell
npm install
npm run dev
```

Open the local URL printed by Next.js.

## What To Inspect

- Current score components and decision output.
- Static or dynamic threshold state.
- Preference-probability decisions when a preference layer is active.
- Logged profile history and screenshots for later labeling or error analysis.

The Python automation scripts own scoring and logging. This dashboard is an observer over those artifacts, not the training entrypoint.
