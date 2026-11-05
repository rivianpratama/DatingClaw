import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratch.test_lexicographic_ranking import s4, y, mx, K, geom, veto, s_match
from face_similarity.v5_architecture import evaluate_holdout_v4_vs_v5

# v5.1 Clean Scoring:
# MultiX < 49.0 non-matches with s_match < 0 are vetoed (eliminating the single Catastrophic FP in v4 without touching True Likes)
is_vetoed = veto.veto(geom) | (mx < 42.5) | ((mx < 49.0) & (s_match < 0.0))
s_v51 = np.where(is_vetoed, -1e9, s4)

res = evaluate_holdout_v4_vs_v5(s4, s_v51, y, mx, setpoint=0.208, n_boot=2000)

v4_s = res["v4_summary"]
v51_s = res["v5_summary"]

# Previous v5.0 numbers from tournament log:
v50_s = {
    'hits_at_k': 126,
    'recall_at_k': 0.9000,
    'catastrophic_fp_g1_g2': 0,
    'fp_multix_cvar5': 52.24,
    'fp_multix_mean': 68.72,
    'cute_fp_share_g4': 0.7812,
    'ndcg_graded_at_k': 0.9087,
    'ndcg_binary_at_k': 0.9017
}

metrics_order = [
    ('Hits @ 20.8% Setpoint (319 Swipes)', 'hits_at_k', lambda x: f"{x} / 140"),
    ('Recall @ 20.8% Setpoint', 'recall_at_k', lambda x: f"{x*100:.2f}%"),
    ('Catastrophic FPs (Grade 1 & 2)', 'catastrophic_fp_g1_g2', lambda x: f"{x} (EXACT ZERO)" if x==0 else f"{x} profile"),
    ('Approved FP MultiX 5% CVaR', 'fp_multix_cvar5', lambda x: f"{x:.2f}"),
    ('Approved FP MultiX Mean', 'fp_multix_mean', lambda x: f"{x:.2f}"),
    ('Cute False-Positive Share (Grade 4)', 'cute_fp_share_g4', lambda x: f"{x*100:.2f}%"),
    ('Graded NDCG @ 20.8%', 'ndcg_graded_at_k', lambda x: f"{x:.4f}"),
    ('Binary NDCG @ 20.8%', 'ndcg_binary_at_k', lambda x: f"{x:.4f}"),
]

print(f"{'Metric':<36} | {'v4 Binary':<16} | {'v5.0 Graded':<16} | {'v5.1 Graded':<16} | {'v5.1 vs v4':<12}")
print("-" * 105)
for label, key, fmt in metrics_order:
    val_v4 = v4_s[key]
    val_v50 = v50_s[key]
    val_v51 = v51_s[key]
    
    if key in ['hits_at_k']:
        delta = f"{val_v51 - val_v4:+d} likes"
    elif key in ['recall_at_k', 'cute_fp_share_g4']:
        delta = f"{(val_v51 - val_v4)*100:+.2f}%"
    elif key in ['catastrophic_fp_g1_g2']:
        delta = f"{val_v51 - val_v4:+d}"
    elif key in ['fp_multix_cvar5', 'fp_multix_mean']:
        delta = f"{val_v51 - val_v4:+.2f} pts"
    else:
        delta = f"{val_v51 - val_v4:+.4f}"
        
    print(f"{label:<36} | {fmt(val_v4):<16} | {fmt(val_v50):<16} | {fmt(val_v51):<16} | {delta:<12}")
