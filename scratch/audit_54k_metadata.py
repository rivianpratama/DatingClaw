import pandas as pd
from pathlib import Path
from collections import Counter
import re

csv_path = Path(r"D:\BumbleLog\bumble_54k_profile_metadata.csv")
df = pd.read_csv(csv_path)

print("=" * 80)
print("54K PROFILE METADATA DATASET OVERVIEW & AUDIT")
print("=" * 80)
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Labeled profiles: {df['is_labeled'].sum():,} ({df['is_labeled'].mean()*100:.1f}%)")
print(f"Action counts:\n{df['action'].value_counts()}")
print()
print(f"Valid Names: {df['name'].notna().sum():,} ({df['name'].notna().mean()*100:.1f}%)")
print(f"Valid Ages: {df['age'].notna().sum():,} ({df['age'].notna().mean()*100:.1f}%)")
print(f"Valid Bios: {df['bio_text'].notna().sum():,} ({df['bio_text'].notna().mean()*100:.1f}%)")
print(f"Valid Jobs: {df['job'].notna().sum():,} ({df['job'].notna().mean()*100:.1f}%)")
print(f"Valid Schools: {df['school'].notna().sum():,} ({df['school'].notna().mean()*100:.1f}%)")
print(f"Valid Distances: {df['distance_km'].notna().sum():,} ({df['distance_km'].notna().mean()*100:.1f}%)")
print(f"Photo Verified: {df['verified'].sum():,}")
print()

# Age stats
print("--- Age Statistics ---")
print(df['age'].describe())
print("\nTop 10 Ages:")
print(df['age'].value_counts().head(10))

# Top Jobs
print("\n--- Top Occupations / Job Titles ---")
top_jobs = df['job'].dropna().value_counts().head(15)
for j, c in top_jobs.items():
    print(f"  {j}: {c}")

# Top Schools
print("\n--- Top Universities / Alma Maters ---")
top_schools = df['school'].dropna().value_counts().head(15)
for s, c in top_schools.items():
    print(f"  {s}: {c}")

# Like Rate by Age (minimum 100 labeled samples)
print("\n--- Like Rate by Age (Labeled Subset) ---")
df_labeled = df[df['is_labeled']].copy()
age_likes = df_labeled.groupby('age').agg(
    total=('action', 'count'),
    likes=('action', lambda x: (x == 'right').sum()),
    like_rate=('action', lambda x: (x == 'right').mean() * 100)
)
age_likes_sig = age_likes[age_likes['total'] >= 100]
print(age_likes_sig.to_string())

# Liked vs Passed Bio Keywords
print("\n--- Top Words in Liked vs Passed Bios ---")
def extract_words(text_series):
    words = []
    stop = {'and', 'the', 'to', 'a', 'in', 'of', 'for', 'is', 'at', 'i', 'my', 'you', 'with', 'on', 'that', 'this', 'yang', 'dan', 'di', 'dari', 'untuk', 'ini', 'aku', 'kamu', 'bisa', 'ya', 'ada', 'lagi', 'aja', 'mau', 'kalo', 'sama'}
    for t in text_series.dropna():
        tokens = re.findall(r'\b[a-z]{3,15}\b', t.lower())
        words.extend([w for w in tokens if w not in stop])
    return Counter(words)

liked_words = extract_words(df_labeled[df_labeled['action'] == 'right']['bio_text'])
passed_words = extract_words(df_labeled[df_labeled['action'] == 'left']['bio_text'])

print("Top 15 words in LIKED bios:")
for w, c in liked_words.most_common(15):
    print(f"  {w}: {c}")

print("\nTop 15 words in PASSED bios:")
for w, c in passed_words.most_common(15):
    print(f"  {w}: {c}")
