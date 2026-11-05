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

# Rolling 1,000 profile statistics
df_scores['rolling_bot_rate'] = df_scores['bot_swipe_right'].rolling(1000, min_periods=500).mean() * 100.0
df_scores['rolling_human_like_rate'] = df_scores['human_like'].rolling(1000, min_periods=500).mean() * 100.0
df_scores['rolling_mean_score'] = df_scores['score'].rolling(1000, min_periods=500).mean()

# Lead-lag cross correlation
corrs = {}
for lag in [-2000, -1000, -500, 0, 500, 1000, 2000]:
    s_lag = df_scores['rolling_bot_rate'].shift(lag)
    c = df_scores['rolling_human_like_rate'].corr(s_lag)
    corrs[lag] = c

print("=" * 80)
print("LEAD-LAG CORRELATIONS (Rolling Bot Swipe Rate vs Human Like Rate)")
print("=" * 80)
for lag, c in corrs.items():
    direction = "Lag (Bot rate at t-lag -> Quality at t)" if lag > 0 else ("Lead (Quality at t -> Bot rate at t+lead)" if lag < 0 else "Synchronous")
    print(f"  Shift {lag:+5d} cards: corr = {c:+.4f}  [{direction}]")

# Fine-grained 2% bucket analysis of bot right-swipe rate
print("\n" + "=" * 80)
print("FINE-GRAINED 2% BUCKETS OF BOT SWIPE RATE VS INCOMING HUMAN LIKE RATE")
print("=" * 80)
fine_bins = [0, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 40, 100]
fine_labels = ['<10%', '10-12%', '12-14%', '14-16%', '16-18%', '18-20%', '20-22%', '22-24%', '24-26%', '26-28%', '28-30%', '30-40%', '>40%']
df_scores['rate_bucket'] = pd.cut(df_scores['rolling_bot_rate'], bins=fine_bins, labels=fine_labels)

bucket_stats = df_scores.groupby('rate_bucket', observed=False).agg(
    n_cards=('screenshot', 'count'),
    mean_human_like_pct=('human_like', lambda x: x.mean() * 100.0),
    human_likes=('human_like', 'sum'),
    mean_threshold=('threshold', 'mean')
).reset_index()

print(bucket_stats.to_string(index=False))
