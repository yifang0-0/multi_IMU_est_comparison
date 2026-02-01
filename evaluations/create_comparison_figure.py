"""Generate method comparison barplot from RMSE results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Methods to include and their display names
METHODS = {
    'Madgwick': 'Madgwick',
    'VQF-OpenSim': 'VQF-OpenSim',
    'vqf+olsson+heading_correction': 'VQF+Olsson+HC',
    'kf_gframe_optimized': 'KF GFrame Opt',
}

knee_df = pd.read_csv('results/knee_rmse_summary.csv', index_col='subject')
ankle_df = pd.read_csv('results/ankle_rmse_summary.csv', index_col='subject')

# Exclude MEAN row to get per-subject values
knee_df = knee_df.drop('MEAN')
ankle_df = ankle_df.drop('MEAN')

display_names = list(METHODS.values())
knee_median = [np.median(knee_df[col]) for col in METHODS.keys()]
ankle_median = [np.median(ankle_df[col]) for col in METHODS.keys()]

# Create grouped barplot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(display_names))
width = 0.35

ax.bar(x - width/2, knee_median, width, label='Knee', color='#3498db')
ax.bar(x + width/2, ankle_median, width, label='Ankle', color='#e67e22')

ax.set_ylabel('Median RMSE (degrees)', fontsize=13, fontweight='bold')
ax.set_title('Joint Angle Estimation Method Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('plots/method_comparison.pdf', bbox_inches='tight')
print("Method comparison plot saved to plots/method_comparison.pdf")
plt.close()
