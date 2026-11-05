import pandas as pd
import numpy as np

df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")

df_labels['human_like'] = (df_labels['rating'] >= 50.0).astype(int)
labels_map = dict(zip(df_labels['filename'], df_labels['human_like']))
df_scores['human_like'] = df_scores['screenshot'].map(labels_map)
df_scores['bot_swipe_right'] = (df_scores['action'] == 'right').astype(int)

# Check unique setup_name, method, dynamic settings
print("Setup names:")
print(df_scores['setup_name'].value_counts(dropna=False))

print("\nMethods:")
print(df_scores['method'].value_counts(dropna=False))

print("\nDynamic targets:")
print(df_scores['dynamic_target_right_rate'].value_counts(dropna=False))

print("\nThreshold statistics:")
print(df_scores['threshold'].describe())
