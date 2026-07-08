# 7k Dataset Methodology

This document describes how the existing 7,196-image rating dataset is packaged for remote labeling studies without publishing private training artifacts.

## Dataset Shape

The source labels live in `dataset_labels.csv` and use the existing rating schema:

```csv
path,rating_1_5,rating
references/example.jpg,3,50
```

The score mapping is fixed:

```text
1 = 0
2 = 25
3 = 50
4 = 75
5 = 100
```

The local 7k label distribution is:

| Rating | Count |
| --- | ---: |
| 1 | 679 |
| 2 | 2,536 |
| 3 | 1,865 |
| 4 | 1,605 |
| 5 | 511 |

## Packaging

Run the packager from the repo root:

```powershell
python package_7k_dataset.py --overwrite
```

By default it writes `D:\BumbleLabel` with:

- `images\bc7k_000001.ext` style flat image filenames.
- `labels.csv` containing `image_id,filename,original_path,rating_1_5,rating`.
- `manifest.json` containing row counts, label distribution, source CSV hash, generation time, and blind remote-labeling tasks.

The packager renames files because the original dataset contains duplicate basenames that would collide in one flat directory.

## Source Requirements

Only include images that were procured with permission for the intended study. Do not mix participant-provided screenshots, dating-app logs, browser state, embeddings, fitted models, or private validation tables into this package.

## Remote Labeling Boundary

The remote labeling app is designed for invite-auth access. Images should remain in a private Cloudflare R2 bucket and be served through short-lived signed URLs. Participant labels should be stored separately from the seed labels under participant-specific paths.

Do not make the R2 bucket public unless every image in the study has explicit public-release clearance.
