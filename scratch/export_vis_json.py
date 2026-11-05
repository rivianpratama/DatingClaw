import pandas as pd
import json
from pathlib import Path

csv_path = Path(r"D:\BumbleLog\bumble_54k_profile_metadata.csv")
df = pd.read_csv(csv_path)

labeled = df[df['is_labeled']].copy()
base_rate = float((labeled['action'] == 'right').mean() * 100)

# 1. Age distribution & Like rate
age_df = labeled.groupby('age').agg(
    total=('action', 'count'),
    likes=('action', lambda x: (x == 'right').sum()),
    rate=('action', lambda x: (x == 'right').mean() * 100)
).reset_index()
age_df = age_df[(age_df['age'] >= 19) & (age_df['age'] <= 35)]

age_data = {
    'labels': [int(a) for a in age_df['age']],
    'totals': [int(t) for t in age_df['total']],
    'likes': [int(l) for l in age_df['likes']],
    'rates': [round(float(r), 1) for r in age_df['rate']]
}

# 2. Key Signal comparison (Relative % boost/penalty vs baseline)
signals = [
    {"label": "Binus University", "rate": 17.5, "count": 137, "diff": +58.4, "type": "pos"},
    {"label": "Travel Bio/Badge", "rate": 16.3, "count": 363, "diff": +46.9, "type": "pos"},
    {"label": "Catholic Religion", "rate": 15.6, "count": 1154, "diff": +41.0, "type": "pos"},
    {"label": "ITB Bandung", "rate": 14.3, "count": 77, "diff": +29.2, "type": "pos"},
    {"label": "Undergraduate Degree", "rate": 14.0, "count": 1931, "diff": +26.9, "type": "pos"},
    {"label": "Baseline Average", "rate": round(base_rate, 1), "count": len(labeled), "diff": 0.0, "type": "neutral"},
    {"label": "Finance / Banking", "rate": 7.3, "count": 383, "diff": -33.9, "type": "neg"},
    {"label": "Teacher / Edu", "rate": 5.2, "count": 268, "diff": -52.8, "type": "neg"},
    {"label": "BUMN State Corp", "rate": 4.8, "count": 42, "diff": -56.9, "type": "neg"},
    {"label": "Nurse / Health", "rate": 3.3, "count": 123, "diff": -70.6, "type": "neg"},
    {"label": "Age 33+ Cohort", "rate": 2.8, "count": 379, "diff": -74.7, "type": "neg"}
]

# 3. Overall numbers
summary = {
    'total_profiles': len(df),
    'labeled_profiles': len(labeled),
    'right_swipes': int((labeled['action'] == 'right').sum()),
    'left_swipes': int((labeled['action'] == 'left').sum()),
    'base_rate': round(base_rate, 2),
    'names_extracted': int(df['name'].notna().sum()),
    'ages_extracted': int(df['age'].notna().sum()),
    'bios_extracted': int(df['bio_text'].notna().sum()),
    'jobs_extracted': int(df['job'].notna().sum()),
    'schools_extracted': int(df['school'].notna().sum())
}

output = {
    'summary': summary,
    'age_data': age_data,
    'signals': signals
}

with open(r"C:\Users\Rivian\.gemini\antigravity\brain\73824904-181e-4277-ac0d-85aa19274479\metadata_vis_data.json", 'w') as f:
    json.dump(output, f, indent=2)

print("Exported metadata_vis_data.json successfully!")
