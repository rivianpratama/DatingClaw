import pandas as pd
import numpy as np

df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")

def parse_ts(ts_str):
    try:
        return pd.to_datetime(str(ts_str)[:15], format='%Y%m%d_%H%M%S', errors='coerce')
    except Exception:
        return pd.NaT

df_scores['dt'] = df_scores['timestamp'].apply(parse_ts)
df_labels['human_like'] = (df_labels['rating'] >= 50.0).astype(int)
labels_map = dict(zip(df_labels['filename'], df_labels['human_like']))
df_scores['human_like'] = df_scores['screenshot'].map(labels_map)
df_scores['bot_swipe_right'] = (df_scores['action'] == 'right').astype(int)
df_scores = df_scores.sort_values('dt').reset_index(drop=True)

# Exclude initial 3,000 cards (new account boost)
df_steady = df_scores.iloc[3000:].copy()

df_steady['rolling_bot_rate'] = df_steady['bot_swipe_right'].rolling(1000, min_periods=500).mean() * 100.0

fine_bins = [0, 12, 16, 18, 20, 22, 24, 26, 30, 100]
fine_labels = ['<12%', '12-16%', '16-18%', '18-20%', '20-22%', '22-24%', '24-26%', '26-30%', '>30%']
df_steady['rate_bucket'] = pd.cut(df_steady['rolling_bot_rate'], bins=fine_bins, labels=fine_labels)

steady_stats = df_steady.groupby('rate_bucket', observed=False).agg(
    n_cards=('screenshot', 'count'),
    labeled_cards=('human_like', lambda x: x.notna().sum()),
    mean_human_like_pct=('human_like', lambda x: x.mean() * 100.0),
    human_likes=('human_like', 'sum'),
    mean_score=('score', 'mean'),
    mean_threshold=('threshold', 'mean'),
    top_tier_profiles_pct=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print("=" * 110)
print("STEADY-STATE ELO DYNAMICS (EXCLUDING INITIAL 3,000 NEW-ACCOUNT CARDS)")
print("=" * 110)
print(steady_stats.to_string(index=False))
