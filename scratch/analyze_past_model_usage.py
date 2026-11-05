import pandas as pd

df = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"].str.slice(0, 15), format="%Y%m%d_%H%M%S", errors="coerce")
N = len(df)

df["regressor_short"] = df["regressor_path"].fillna("none").apply(lambda p: p.split("\\")[-1].replace(".joblib", ""))
df["multi_short"] = df["multimodal_regressor_path"].fillna("none").apply(lambda p: p.split("\\")[-1].replace(".joblib", ""))
df["pref_short"] = df["preference_model_path"].fillna("none").apply(lambda p: p.split("\\")[-1].replace(".joblib", ""))
df["method_clean"] = df["method"].fillna("unknown")

# Pipeline combination
df["pipeline"] = df["method_clean"] + " + [" + df["regressor_short"] + "] + [" + df["pref_short"] + "]"

grouped = []
for name, g in df.groupby("pipeline"):
    t_min = str(g["timestamp"].min())[:10]
    t_max = str(g["timestamp"].max())[:10]
    count = len(g)
    right_swipes = int((g["action"] == "right").sum())
    rate = right_swipes / count * 100.0 if count else 0.0
    grouped.append({
        "pipeline": name,
        "count": count,
        "share": count / N * 100.0,
        "right_swipes": right_swipes,
        "right_rate": rate,
        "start": t_min,
        "end": t_max,
        "method": g["method_clean"].iloc[0],
        "regressor": g["regressor_short"].iloc[0],
        "preference": g["pref_short"].iloc[0],
    })

res = pd.DataFrame(grouped).sort_values("count", ascending=False)
print(f"Total historical swipes in log: {N}\n")
for _, r in res.iterrows():
    print(f"{r['count']:>6} swipes ({r['share']:>4.1f}%) | Right Rate: {r['right_rate']:>4.1f}% | Dates: {r['start']} to {r['end']} | {r['pipeline']}")
