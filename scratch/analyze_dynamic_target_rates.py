import pandas as pd
import numpy as np

# Load scores and labels
df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")

df_labels['human_like'] = (df_labels['rating'] >= 50.0).astype(int)
labels_map = dict(zip(df_labels['filename'], df_labels['human_like']))
df_scores['human_like'] = df_scores['screenshot'].map(labels_map)
df_scores['bot_swipe_right'] = (df_scores['action'] == 'right').astype(int)

# Group by dynamic_target_right_rate
print("=" * 100)
print("BUMBLE REACTION BY EXPLICIT DYNAMIC TARGET RIGHT RATE SETTING")
print("=" * 100)

target_groups = df_scores.groupby('dynamic_target_right_rate').agg(
    total_profiles=('screenshot', 'count'),
    realized_bot_swipe_rate=('bot_swipe_right', lambda x: x.mean() * 100.0),
    labeled_profiles=('human_like', lambda x: x.notna().sum()),
    human_like_rate=('human_like', lambda x: x.mean() * 100.0),
    human_likes_total=('human_like', lambda x: x.sum()),
    mean_profile_score=('score', 'mean'),
    median_profile_score=('score', 'median'),
    mean_threshold=('threshold', 'mean'),
    min_threshold=('threshold', 'min'),
    max_threshold=('threshold', 'max'),
    high_tier_profiles_pct=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print(target_groups.sort_values('dynamic_target_right_rate').to_string(index=False))

# Now let's analyze by OPERATIONAL THRESHOLD BINS
print("\n" + "=" * 100)
print("INCOMING PROFILE QUALITY BY THRESHOLD BINS")
print("=" * 100)

df_scores['thresh_bin'] = pd.cut(df_scores['threshold'], bins=[40, 52, 54, 56, 58, 60, 62, 65, 80])
thresh_groups = df_scores.groupby('thresh_bin', observed=False).agg(
    total_profiles=('screenshot', 'count'),
    realized_bot_swipe_rate=('bot_swipe_right', lambda x: x.mean() * 100.0),
    labeled_profiles=('human_like', lambda x: x.notna().sum()),
    human_like_rate=('human_like', lambda x: x.mean() * 100.0),
    mean_score=('score', 'mean'),
    top_tier_pct=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print(thresh_groups.to_string(index=False))
