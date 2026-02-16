"""Generate method comparison barplot from RMSE results using mean."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Methods to include and their display names
METHODS = {
    'MADGWICK': 'Madgwick + IK',
    'VQF-IK': 'VQF + IK',
    'vqf+olsson+heading_correction': 'VQF + Olsson',
    'kf_gframe_model': 'KF GFrame',
}

knee_df = pd.read_csv('results/knee_rmse_summary.csv', index_col='subject')
ankle_df = pd.read_csv('results/ankle_rmse_summary.csv', index_col='subject')

# Exclude MEAN row to get per-subject values
knee_df = knee_df.drop('MEAN')
ankle_df = ankle_df.drop('MEAN')

display_names = list(METHODS.values())
knee_mean = [np.mean(knee_df[col]) for col in METHODS.keys()]
ankle_mean = [np.mean(ankle_df[col]) for col in METHODS.keys()]

# Create grouped barplot
Y_MAX = 20
fig, ax = plt.subplots(figsize=(10, 3))
x = np.arange(len(display_names))
width = 0.35

# Clip values for plotting
knee_plot = [min(v, Y_MAX) for v in knee_mean]
ankle_plot = [min(v, Y_MAX) for v in ankle_mean]

ax.bar(x - width/2, knee_plot, width, label='Knee', color='#3498db')
ax.bar(x + width/2, ankle_plot, width, label='Ankle', color='#e67e22')

# Mark truncated bars with zigzag break and value labels
def draw_break(ax, bar_x, bar_width, y_break, color):
    """Draw a zigzag break pattern across a bar."""
    half_w = bar_width / 2
    n_zigs = 5
    xs = np.linspace(bar_x - half_w, bar_x + half_w, n_zigs * 2 + 1)
    ys = [y_break + (0.6 if i % 2 else -0.6) for i in range(len(xs))]
    # White background to "cut" the bar
    ax.fill_between(xs, y_break - 0.8, y_break + 0.8, color='white', zorder=4)
    # Zigzag line
    ax.plot(xs, ys, color=color, linewidth=1.5, zorder=5)

# Add value labels and zigzag breaks for truncated bars
for i, (knee_val, ankle_val) in enumerate(zip(knee_mean, ankle_mean)):
    # Knee bar label
    knee_x = x[i] - width/2
    if knee_val > Y_MAX:
        draw_break(ax, knee_x, width, Y_MAX - 1.5, '#3498db')
    label_y = min(knee_val, Y_MAX) + 0.3
    ax.text(knee_x, label_y, f'{knee_val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Ankle bar label
    ankle_x = x[i] + width/2
    if ankle_val > Y_MAX:
        draw_break(ax, ankle_x, width, Y_MAX - 1.5, '#e67e22')
    label_y = min(ankle_val, Y_MAX) + 0.3
    ax.text(ankle_x, label_y, f'{ankle_val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Mean RMSE (degrees)', fontsize=13, fontweight='bold')
ax.set_title('Joint Angle Estimation Method Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=11)
ax.tick_params(axis='x', length=0)  # Hide tick marks
ax.set_xticklabels([])  # Remove x labels
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, Y_MAX)

plt.tight_layout()
plt.savefig('plots/method_comparison.pdf', bbox_inches='tight')
plt.savefig('plots/method_comparison.svg', bbox_inches='tight')
print("Method comparison plot saved to plots/method_comparison.pdf")
plt.close()
