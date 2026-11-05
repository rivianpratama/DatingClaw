import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

# Data
models = ['Legacy (7k)', 'Exp3 (8k)', '10k Pure', '20k Face', '20k Blend', '20k Multi']
dataset_sizes = [7.5, 8.5, 10.0, 20.0, 20.0, 20.0]
pr_auc = [0.3142, 0.3151, 0.3514, 0.3633, 0.4534, 0.4576]
junk_filtered = [30.5, 30.5, 42.9, 41.8, 53.1, 53.1]
colors = ['#888888', '#888888', '#3498db', '#e67e22', '#2ecc71', '#9b59b6']

# Subplot 1: PR-AUC
bars1 = ax1.bar(models, pr_auc, color=colors, width=0.6, edgecolor='black', alpha=0.85)
ax1.set_title('Ranking Precision (PR-AUC) on Untrained 20k-22k Holdout\n(Higher is Better)', fontsize=12, fontweight='bold', pad=12)
ax1.set_ylabel('PR-AUC Score', fontsize=11)
ax1.set_ylim(0, 0.55)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Annotations on bars1
for bar, val in zip(bars1, pr_auc):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add trend arrow from 10k to 20k Blend
ax1.annotate('+29.0% Accuracy Jump',
             xy=(2, 0.355), xytext=(4, 0.49),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', color='#1e8449', ha='center')

# Subplot 2: Junk Filtered at 95% Recall (Safe Bouncer)
bars2 = ax2.bar(models, junk_filtered, color=colors, width=0.6, edgecolor='black', alpha=0.85)
ax2.set_title('Safe Bouncer: Junk Bypassed at ≥95% Like Retention\n(Higher is Better - Zero Missed Likes Target)', fontsize=12, fontweight='bold', pad=12)
ax2.set_ylabel('% of Junk Discarded Without Viewing', fontsize=11)
ax2.set_ylim(0, 70)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Annotations on bars2
for bar, val in zip(bars2, junk_filtered):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Reference baseline label
ax2.axhline(0, color='red', linestyle='-', linewidth=1)
ax2.annotate('Bypasses 53% of all unwanted cards\nwhile saving 150 of 157 likes',
             xy=(4, 54), xytext=(3.5, 62),
             arrowprops=dict(facecolor='#1e8449', shrink=0.08, width=1.5, headwidth=8),
             fontsize=9.5, fontweight='bold', color='#1e8449', ha='center')

plt.tight_layout()

# Save
output_path1 = r'C:\Users\Rivian\.gemini\antigravity\brain\73824904-181e-4277-ac0d-85aa19274479\scaling_benchmark_chart.png'
output_path2 = r'c:\Users\Rivian\Documents\GitHub\BumbleClaw\results\scaling_benchmark_chart.png'
fig.savefig(output_path1, dpi=200, bbox_inches='tight')
fig.savefig(output_path2, dpi=200, bbox_inches='tight')
print("Successfully saved scaling benchmark chart!")
