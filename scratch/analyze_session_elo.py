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

# Identify sessions based on gap > 20 minutes
df_scores['time_diff'] = df_scores['dt'].diff().dt.total_seconds()
df_scores['new_session'] = (df_scores['time_diff'] > 1200) | (df_scores['time_diff'].isna())
df_scores['session_id'] = df_scores['new_session'].cumsum()

session_stats = df_scores.groupby('session_id').agg(
    start_time=('dt', 'min'),
    end_time=('dt', 'max'),
    duration_min=('time_diff', lambda x: x.sum() / 60.0),
    cards=('screenshot', 'count'),
    bot_right_rate=('bot_swipe_right', lambda x: x.mean() * 100.0),
    labeled_cards=('human_like', lambda x: x.notna().sum()),
    human_like_rate=('human_like', lambda x: x.mean() * 100.0),
    mean_score=('score', 'mean'),
    mean_threshold=('threshold', 'mean')
).reset_index()

# Filter sessions with at least 100 cards
large_sessions = session_stats[session_stats['cards'] >= 100].copy()
large_sessions['next_human_like_rate'] = large_sessions['human_like_rate'].shift(-1)
large_sessions['next_mean_score'] = large_sessions['mean_score'].shift(-1)

print("=" * 100)
print(f"Total sessions: {len(session_stats)} | Sessions with >= 100 cards: {len(large_sessions)}")
print("=" * 100)
print("\nTop 15 Largest Sessions:")
print(large_sessions[['session_id', 'start_time', 'cards', 'bot_right_rate', 'human_like_rate', 'mean_threshold']].head(15).to_string(index=False))

# Group large sessions by bot_right_rate bins
bins = [0, 15, 18, 21, 25, 30, 100]
bin_labels = ['< 15%', '15% - 18%', '18% - 21%', '21% - 25%', '25% - 30%', '> 30%']
large_sessions['rate_bin'] = pd.cut(large_sessions['bot_right_rate'], bins=bins, labels=bin_labels)

sess_grouped = large_sessions.groupby('rate_bin', observed=False).agg(
    n_sessions=('session_id', 'count'),
    total_cards=('cards', 'sum'),
    avg_cards_per_sess=('cards', 'mean'),
    current_human_like_rate=('human_like_rate', 'mean'),
    next_session_human_like_rate=('next_human_like_rate', 'mean'),
    avg_score=('mean_score', 'mean'),
    avg_threshold=('mean_threshold', 'mean')
).reset_index()

print("\n" + "=" * 100)
print("SESSION-LEVEL ELO TRANSFER FUNCTION (CURRENT SESSION SWIPE RATE -> NEXT SESSION QUALITY)")
print("=" * 100)
print(sess_grouped.to_string(index=False))
