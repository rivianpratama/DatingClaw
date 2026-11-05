import pandas as pd
import numpy as np

# 1. Load data
print("Loading data...")
df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")

# Parse timestamps
def parse_ts(ts_str):
    try:
        # e.g. 20260519_154255_550965
        return pd.to_datetime(str(ts_str)[:15], format='%Y%m%d_%H%M%S', errors='coerce')
    except Exception:
        return pd.NaT

df_scores['dt'] = df_scores['timestamp'].apply(parse_ts)

# Prepare binary label mapping
df_labels['human_like'] = (df_labels['rating'] >= 50.0).astype(int)
labels_map = dict(zip(df_labels['filename'], df_labels['human_like']))

df_scores['human_like'] = df_scores['screenshot'].map(labels_map)
df_scores['bot_swipe_right'] = (df_scores['action'] == 'right').astype(int)

# Sort by chronological order
df_scores = df_scores.sort_values('dt').reset_index(drop=True)

N = len(df_scores)
print(f"Total scored profiles: {N}")
print(f"Date range: {df_scores['dt'].min()} to {df_scores['dt'].max()}")
print(f"Total labeled profiles: {df_scores['human_like'].notna().sum()} / {N}")
print(f"Overall Bot Right Swipe Rate: {df_scores['bot_swipe_right'].mean()*100:.2f}%")
print(f"Overall Human Like Rate (ground truth): {df_scores['human_like'].mean()*100:.2f}%")

# Rolling windows: 250, 500, 1000 cards
for w in [250, 500, 1000]:
    df_scores[f'bot_rate_{w}'] = df_scores['bot_swipe_right'].rolling(window=w, min_periods=w//2).mean() * 100.0
    df_scores[f'human_rate_{w}'] = df_scores['human_like'].rolling(window=w, min_periods=w//2).mean() * 100.0
    df_scores[f'mean_score_{w}'] = df_scores['score'].rolling(window=w, min_periods=w//2).mean()

# Analyze by binned bot right-swipe rate
# Let's see: how does the bot's right-swipe rate in the PREVIOUS window correlate with the NEXT window's incoming quality?
print("\n" + "=" * 90)
print("1. DIRECT REACTION: BUCKETED BOT RIGHT-SWIPE RATE VS. INCOMING PROFILE ATTRACTIVENESS")
print("=" * 90)

# We define lag: bot swipe rate in window [t-500 : t] vs incoming profile quality in [t : t+500]
df_scores['bot_rate_prev_500'] = df_scores['bot_swipe_right'].shift(500).rolling(window=500, min_periods=250).mean() * 100.0
df_scores['human_rate_next_500'] = df_scores['human_like'].rolling(window=500, min_periods=250).mean() * 100.0
df_scores['mean_score_next_500'] = df_scores['score'].rolling(window=500, min_periods=250).mean()

# Clean df for analysis
df_clean = df_scores.dropna(subset=['bot_rate_prev_500', 'human_rate_next_500', 'mean_score_next_500']).copy()

bins = [0, 8, 12, 16, 20, 25, 30, 40, 100]
labels = ['< 8%', '8% - 12%', '12% - 16%', '16% - 20%', '20% - 25%', '25% - 30%', '30% - 40%', '> 40%']
df_clean['bot_rate_bin'] = pd.cut(df_clean['bot_rate_prev_500'], bins=bins, labels=labels)

grouped = df_clean.groupby('bot_rate_bin', observed=False).agg(
    count=('screenshot', 'count'),
    mean_human_like_rate=('human_rate_next_500', 'mean'),
    median_human_like_rate=('human_rate_next_500', 'median'),
    mean_bot_score=('mean_score_next_500', 'mean'),
    median_bot_score=('mean_score_next_500', 'median')
).reset_index()

print(grouped.to_string(index=False))

# Now let's do instantaneous rolling correlation and 1,000-card lag analysis
print("\n" + "=" * 90)
print("2. 1,000-CARD LAGGED ELO ANALYSIS (PREVIOUS 1,000 SWIPES -> NEXT 1,000 PROFILES)")
print("=" * 90)

df_scores['bot_rate_prev_1000'] = df_scores['bot_swipe_right'].shift(1000).rolling(window=1000, min_periods=500).mean() * 100.0
df_scores['human_rate_next_1000'] = df_scores['human_like'].rolling(window=1000, min_periods=500).mean() * 100.0
df_scores['mean_score_next_1000'] = df_scores['score'].rolling(window=1000, min_periods=500).mean()

df_clean_1k = df_scores.dropna(subset=['bot_rate_prev_1000', 'human_rate_next_1000', 'mean_score_next_1000']).copy()
df_clean_1k['bot_rate_bin'] = pd.cut(df_clean_1k['bot_rate_prev_1000'], bins=bins, labels=labels)

grouped_1k = df_clean_1k.groupby('bot_rate_bin', observed=False).agg(
    count=('screenshot', 'count'),
    incoming_human_like_pct=('human_rate_next_1000', 'mean'),
    incoming_mean_score=('mean_score_next_1000', 'mean'),
    incoming_top10_density=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print(grouped_1k.to_string(index=False))

# Now let's analyze by Era / Historical Segment
print("\n" + "=" * 90)
print("3. ERA PROGRESSION (EVERY 5,000 PROFILES ACROSS 54K HISTORY)")
print("=" * 90)

df_scores['era_chunk'] = (df_scores.index // 5000) * 5000
era_grouped = df_scores.groupby('era_chunk').agg(
    total_profiles=('screenshot', 'count'),
    bot_swipe_rate=('bot_swipe_right', lambda x: x.mean() * 100.0),
    human_like_rate=('human_like', lambda x: x.mean() * 100.0),
    mean_model_score=('score', 'mean'),
    top_tier_pct=('score', lambda x: (x >= 70.0).mean() * 100.0)
).reset_index()

print(era_grouped.to_string(index=False))
