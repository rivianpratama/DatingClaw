import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from face_similarity.store import load_store

df = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")
print(f"Total rows in binary_swipe_labels: {len(df)}")

df_train = df.iloc[:40000].copy()
df_val = df.iloc[40000:].copy()

n_train_likes = (df_train['rating'] >= 50.0).sum()
n_val_likes = (df_val['rating'] >= 50.0).sum()

print(f"Train split (0..40,000): {len(df_train)} rows | {n_train_likes} Likes ({n_train_likes/len(df_train)*100:.2f}%)")
print(f"Val split (40,000..{len(df)}): {len(df_val)} rows | {n_val_likes} Likes ({n_val_likes/len(df_val)*100:.2f}%)")

# Check existing feature stores coverage
stores = [
    "embeddings/reference_store_bumble_30k_pure.npz",
    "embeddings/reference_store_bumble_30k_to_33k.npz",
    "embeddings/reference_store_bumble_33k_to_40k.npz",
    "embeddings/reference_store_bumble_40k_to_41k.npz"
]

all_paths = set()
for s in stores:
    p = Path(s)
    if p.exists():
        st = load_store(str(p))
        all_paths.update(st.paths)
        print(f"Store {p.name}: {len(st.paths)} paths")

train_missing = [p for p in df_train['path'] if p not in all_paths]
val_missing = [p for p in df_val['path'] if p not in all_paths]

print(f"Train paths missing in cached stores: {len(train_missing)}")
print(f"Val paths missing in cached stores: {len(val_missing)}")
