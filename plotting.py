"""Plotting utilities for joint angle estimation comparison."""
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


def plot_time_series_error(errors_dict, joint_name='Joint', save_path=None, show=True, xlim=(0, 1000), num_entries=4):
    """Plot error time series with RMSE stats."""
    if not errors_dict:
        return
    _, ax = plt.subplots(figsize=(14, 6))
    labels = list(errors_dict.keys())[:num_entries]

    for i, label in enumerate(labels):
        errors = errors_dict[label]
        ax.plot(np.arange(len(errors)), errors, label=label, color=COLORS[i % len(COLORS)],
                alpha=0.8, linewidth=1.0)

    ax.set_title(f'{joint_name} Angle Estimation Error Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sample', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Error (degrees)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    stats_lines = [f"{label}: RMSE={np.sqrt(np.mean(np.array(errors_dict[label])**2)):.2f}"
                   for label in labels]
    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path is None:
        save_path = f'plots/time_series_error_{joint_name.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series error plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_error_comparison(errors_dict, joint_name='Joint', save_path=None, show=True):
    """Plot error distribution boxplot."""
    _, ax = plt.subplots(figsize=(12, 7))
    labels = list(errors_dict.keys())
    data = [errors_dict[label] for label in labels]
    bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False,
                       widths=0.6, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bplot['boxes'], COLORS[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(f'{joint_name} Angle Estimation Error Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Absolute Error (degrees)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    stats_text = '\n'.join([
        f"{labels[i]}: RMSE={np.sqrt(np.mean(np.array(data[i])**2)):.2f} | Mean={np.mean(data[i]):.2f}"
        for i in range(len(labels))
    ])
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.tight_layout()
    if save_path is None:
        save_path = f'plots/error_comparison_{joint_name.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error comparison plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()
