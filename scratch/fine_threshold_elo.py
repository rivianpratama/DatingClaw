import pandas as pd
import numpy as np

df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")

df_labels['human_like'] = (df_labels['rating'] >= 50.0).astype(int)
labels_map = dict(zip(df_labels['filename'], df_labels['human_like']))
df_scores['human_like'] = df_scores['screenshot'].map(labels_map)
df_scores['bot_swipe_right'] = (df_scores['action'] == 'right').astype(int)

# Steady-state post card 3000
df_steady = df_scores.iloc[3000:].copy()

# Threshold bins in increments of 2 points
t_bins = [48, 52, 54, 55, 56, 57, 58, 59, 60, 62, 65, 75]
df_steady['t_bucket'] = pd.cut(df_steady['threshold'], bins=t_bins)

t_stats = df_steady.groupby('t_bucket', observed=False).agg(
    cards=('screenshot', 'count'),
    labeled=('human_like', lambda x: x.notna().sum()),
    realized_swipe_rate=('bot_swipe_right', lambda x: x.mean() * 100.0),
    human_like_rate=('human_like', lambda x: x.mean() * 100.0),
    human_likes_count=('human_like', 'sum'),
    mean_score=('score', 'mean'),
    top_tier_pct=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print("=" * 110)
print("INCOMING PROFILE ATTRACTIVENESS BY LIVE OPERATIONAL THRESHOLD (STEADY STATE)")
print("=" * 110)
print(t_stats.to_string(index=False))
