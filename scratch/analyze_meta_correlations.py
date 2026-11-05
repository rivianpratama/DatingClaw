import pandas as pd
from pathlib import Path

csv_path = Path(r"D:\BumbleLog\bumble_54k_profile_metadata.csv")
df = pd.read_csv(csv_path)
labeled = df[df['is_labeled']].copy()
base_rate = (labeled['action'] == 'right').mean() * 100

print(f"Overall Baseline Like Rate: {base_rate:.2f}% (N = {len(labeled):,})\n")

def test_signal(name, mask):
    subset = labeled[mask]
    n = len(subset)
    if n < 30:
        return
    likes = (subset['action'] == 'right').sum()
    rate = (likes / n) * 100
    rel_diff = ((rate / base_rate) - 1.0) * 100
    print(f"{name:35s} | N={n:5d} | Likes={likes:4d} | Rate={rate:5.1f}% | vs Base: {rel_diff:+5.1f}%")

print("=" * 80)
print("FEATURE LIKE RATE AUDIT (SAMPLE SIZE >= 30)")
print("=" * 80)

# 1. Verification
test_signal("Photo Verified = True", labeled['verified'] == True)
test_signal("Photo Verified = False", labeled['verified'] == False)

# 2. Badges
badges_to_test = ['New here', 'Travel mode', 'Undergraduate degree', 'In college', 'Postgraduate degree']
for b in badges_to_test:
    mask = labeled['badge'].fillna('').str.contains(b, case=False, regex=False)
    test_signal(f"Badge: {b}", mask)

# 3. Specific Keywords in Bio / Job / Raw text
keywords = [
    'catholic', 'kristen', 'christian', 'islam', 'muslim', 'buddha',
    'doctor', 'dokter', 'dentist', 'nurse', 'engineer', 'finance', 'teacher',
    'bank', 'pns', 'bumn', 'ui', 'ugm', 'itb', 'binus', 'dog', 'cat',
    'coffee', 'gym', 'workout', 'travel', 'foodie', 'music', 'art'
]

print("\n--- Keyword Signals in Bio/Profile ---")
for kw in keywords:
    mask = labeled['raw_text'].fillna('').str.contains(rf'\b{kw}\b', case=False, regex=True)
    test_signal(f"Keyword: '{kw}'", mask)
