# Remote Labeling Deployment Guide

This guide deploys the 7k remote labeling system with Vercel for the Next.js app and Cloudflare R2 for private image/task/label storage.

## What Gets Deployed

- Vercel hosts the dashboard app, especially `/label`.
- Cloudflare R2 stores the dataset images, the labeling task manifest, and every submitted score.
- The browser never receives R2 credentials. It receives a short-lived signed image URL from the Vercel API.

The current storage layout is:

```text
datasets/7k/images/bc7k_000001.jpg
datasets/7k/tasks/manifest.json
labels/{participant_id}/{task_id}.json
```

Each score is saved automatically when a participant taps a 1-5 rating button. The API writes one immutable JSON file per participant/task:

```json
{
  "participant_id": "participant_001",
  "task_id": "bc7k_000001",
  "image_id": "bc7k_000001",
  "filename": "bc7k_000001.jpg",
  "rating_1_5": 4,
  "rating": 75,
  "submitted_at": "2026-05-27T00:00:00.000Z",
  "participant_name": "participant_001",
  "user_agent": "..."
}
```

This is the database for v1: R2 object storage, one JSON object per submitted label. It is append-only by task key: duplicate submissions for the same participant/task return `409`.

## 1. Prepare The Local Dataset Package

From the repo root:

```powershell
python package_7k_dataset.py --overwrite
```

The package is written to:

```text
D:\BumbleLabel
```

Expected local files:

```text
D:\BumbleLabel\images\bc7k_000001.jpg
D:\BumbleLabel\labels.csv
D:\BumbleLabel\manifest.json
```

Keep `labels.csv` private if you want remote participants to label blind. Upload only the images and manifest described below.

## 2. Create Cloudflare R2 Bucket And Credentials

1. In Cloudflare, create an R2 bucket, for example `bumble-label`.
2. Create R2 S3 API credentials with access limited to that bucket.
3. Record:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET`

The app uses Cloudflare R2 through the S3-compatible endpoint:

```text
https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
```

## 3. Upload Dataset Objects To R2

Upload local files into this exact R2 layout:

```text
D:\BumbleLabel\images\*       -> datasets/7k/images/
D:\BumbleLabel\manifest.json  -> datasets/7k/tasks/manifest.json
```

Do not upload `labels.csv` for participant-facing blind labeling unless you intentionally want the ground-truth labels in the bucket.

If using AWS CLI with an R2 profile, the shape is:

```powershell
aws s3 sync D:\BumbleLabel\images s3://bumble-label/datasets/7k/images --endpoint-url https://945caa8c97d24f7178be34592cdae03f.r2.cloudflarestorage.com/bumble-label
aws s3 cp D:\BumbleLabel\manifest.json s3://bumble-label/datasets/7k/tasks/manifest.json --endpoint-url https://945caa8c97d24f7178be34592cdae03f.r2.cloudflarestorage.com/bumble-label
```

## 4. Configure R2 CORS

Because the browser loads signed R2 image URLs directly, the bucket needs CORS for the Vercel origin.

Use a CORS policy like this, replacing the origins:

```json
[
  {
    "AllowedOrigins": [
      "https://your-project.vercel.app",
      "https://your-custom-domain.com",
      "http://localhost:3000"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Keep the bucket private. CORS enables browser access to valid signed URLs; it does not make the bucket public.

## 5. Configure Participants

Participant access is controlled by one checked-in JSON file:

```text
dashboard/label_participants.json
```

Add or remove names there:

```json
{
  "participants": [
    "participant_001",
    "participant_002",
    "Alice Study A"
  ]
}
```

The participant types their name on `/label`. Matching is case-insensitive and collapses repeated spaces. Storage paths use a sanitized ID derived from the name, so `Alice Study A` writes to:

```text
labels/Alice_Study_A/{task_id}.json
```

This is a lightweight gate, not strong authentication. Anyone who knows an allowed participant name can enter that queue. For a closed study, use participant aliases that are not easy to guess.

After editing `label_participants.json`, commit and redeploy Vercel so the serverless functions receive the updated file.

## 6. Configure Vercel

Deploy the `dashboard` directory as the Vercel project.

Recommended Vercel project settings:

```text
Framework Preset: Next.js
Root Directory: dashboard
Build Command: npm run build
Install Command: npm install
Output Directory: .next
```

Add these environment variables in Vercel Project Settings:

```text
R2_ACCOUNT_ID=<cloudflare-account-id>
R2_ACCESS_KEY_ID=<r2-access-key>
R2_SECRET_ACCESS_KEY=<r2-secret-key>
R2_BUCKET=bumble-label
R2_DATASET_PREFIX=datasets/7k
R2_LABEL_PREFIX=labels
R2_SIGNED_URL_TTL_SECONDS=900
```

After adding or changing environment variables, redeploy the Vercel project. Vercel does not apply changed env vars to old deployments.

## 7. Participant Flow

Send each participant:

```text
https://your-project.vercel.app/label
```

The participant enters their allowed name once. The browser stores it in `sessionStorage`, requests the next task from `/api/labeling/task`, then submits scores to `/api/labeling/labels`.

## 8. How Scores Are Saved

Every rating button calls:

```text
POST /api/labeling/labels
```

with:

```json
{
  "image_id": "bc7k_000001",
  "filename": "bc7k_000001.jpg",
  "rating_1_5": 4
}
```

The server validates the participant name against `label_participants.json`, checks that the task exists in `datasets/7k/tasks/manifest.json`, maps the rating to the existing score scale, and writes:

```text
labels/participant_001/bc7k_000001.json
```

That R2 path is the durable record. To export the database later, list objects under `labels/`, read each JSON file, and flatten them into CSV with columns such as:

```text
participant_id,participant_name,task_id,image_id,filename,rating_1_5,rating,submitted_at,user_agent
```

## 9. Smoke Test

Before sending real participant links:

1. Open `/label`.
2. Enter a test participant name from `label_participants.json`.
3. Confirm the first image loads.
4. Submit a rating.
5. In R2, confirm this object exists:

```text
labels/<participant-id>/<image-id>.json
```

6. Refresh `/label`; the app should advance to the next unlabeled task.
7. Try a name not in `label_participants.json`; the API should return `401`.

## References

- Cloudflare R2 presigned URLs: https://developers.cloudflare.com/r2/api/s3/presigned-urls/
- Cloudflare R2 CORS: https://developers.cloudflare.com/r2/buckets/cors/
- Vercel Next.js deployment: https://vercel.com/docs/concepts/next.js/overview
- Vercel environment variables: https://vercel.com/docs/projects/environment-variables
