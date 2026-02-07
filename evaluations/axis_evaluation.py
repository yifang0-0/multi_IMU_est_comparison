"""Joint axis evaluation script for comparing axis estimation methods."""
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import qmt
from pathlib import Path
import pandas as pd

from run_estimation import prepare_data
from methods.kf_gframe import estimate_lever_arms
from dfjimu.mekf_acc import mekf_acc
from methods.rnno import compute_rnno_orientation
from methods.shared import calculate_joint_angle
from methods.axis import AXIS_METHODS, OPENSIM_JOINT_AXES
from constants import VALID_SUBJECTS

CACHE_DIR = Path('cache/orientations')


# =============================================================================
# Cache Management
# =============================================================================

def get_cache_path(subject_id, joint):
    """Get cache file path for subject/joint."""
    return CACHE_DIR / f'{subject_id}_{joint}.npz'


def load_cache(
    subject_id,  # subject identifier (e.g., 'Subject08')
    joint,       # 'knee' or 'ankle'
):
    """Load cached orientation data, returns dict with q_rel_kf, q_rel_rnno, gt_angles, imu data, or None."""
    cache_path = get_cache_path(subject_id, joint)
    if not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_cache(subject_id, joint, cache_data):
    """Save orientation data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(subject_id, joint)
    np.savez(cache_path, **cache_data)
    print(f"Saved cache: {cache_path}")


# =============================================================================
# Pre-computation
# =============================================================================

def precompute_orientations(
    subject_id,         # subject identifier (e.g., 'Subject08')
    joint,              # 'knee' or 'ankle'
    include_rnno=True,  # whether to compute RNNO orientation (slower)
):
    """Compute and cache orientations from KF_Gframe and optionally RNNO, returns dict with cached data."""
    print(f"\n{'='*60}")
    print(f"Pre-computing orientations: {subject_id} - {joint}")
    print(f"{'='*60}")

    # Load IMU data using existing prepare_data
    data = prepare_data(joint, subject_id)

    acc_prox = data['acc_prox']
    gyr_prox = data['gyr_prox']
    acc_dist = data['acc_dist']
    gyr_dist = data['gyr_dist']
    fs = data['fs']
    gt = data['gt']

    # Ensure shape is (N, 3) for storage
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # === KF_Gframe orientation ===
    print("\nComputing KF_Gframe orientation...")

    # Estimate lever arms
    r1, r2 = estimate_lever_arms(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)
    print(f"Lever arms: r1={r1}, r2={r2}")

    # Run MEKF-acc
    q1_all, q2_all = mekf_acc(gyr_prox, gyr_dist, acc_prox, acc_dist, r1, r2, fs, np.array([1.0, 0, 0, 0]))

    # Compute relative quaternion
    q_rel_kf = qmt.qmult(qmt.qinv(q1_all), q2_all)
    print(f"KF_Gframe q_rel shape: {q_rel_kf.shape}")  # type: ignore[union-attr]

    cache_data = {
        'q_rel_kf': q_rel_kf,
        'gt_angles': gt,
        'acc_prox': acc_prox,
        'gyr_prox': gyr_prox,
        'acc_dist': acc_dist,
        'gyr_dist': gyr_dist,
        'fs': np.array(fs),
        'r1': r1,
        'r2': r2,
    }

    # === RNNO orientation (optional) ===
    if include_rnno:
        print("\nComputing RNNO orientation...")
        try:
            q_rel_rnno = compute_rnno_orientation(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)
            cache_data['q_rel_rnno'] = q_rel_rnno
            print(f"RNNO q_rel shape: {q_rel_rnno.shape}")
        except Exception as e:
            print(f"RNNO computation failed: {e}")
            print("Continuing without RNNO orientations")

    # Save cache
    save_cache(subject_id, joint, cache_data)

    return cache_data


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_axis_method(
    method_name,        # name of registered method
    cache_data,         # loaded cache dict from load_cache()
    joint='knee',       # 'knee' or 'ankle'
    orientation='kf',   # 'kf' or 'rnno'
):
    """Evaluate a single axis method, returns dict with jhat, angle_deg, rmse, corr."""
    if method_name not in AXIS_METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(AXIS_METHODS.keys())}")

    method_info = AXIS_METHODS[method_name]
    method_func = method_info['func']

    # Select orientation source
    q_rel_key = 'q_rel_rnno' if orientation == 'rnno' else 'q_rel_kf'
    if q_rel_key not in cache_data:
        raise ValueError(f"Orientation '{orientation}' not available in cache")

    q_rel = cache_data[q_rel_key]
    gt_angles = cache_data['gt_angles']

    # Build kwargs
    kwargs = {'q_rel': q_rel, 'joint': joint, 'fs': cache_data.get('fs', 100)}

    if method_info['requires_gt']:
        kwargs['gt_angles'] = gt_angles

    if method_info['requires_imu']:
        kwargs['acc_prox'] = cache_data['acc_prox']
        kwargs['gyr_prox'] = cache_data['gyr_prox']
        kwargs['acc_dist'] = cache_data['acc_dist']
        kwargs['gyr_dist'] = cache_data['gyr_dist']

    # Run method
    jhat = method_func(**kwargs)

    # Calculate angle
    angle_deg = calculate_joint_angle(q_rel, jhat)

    # Handle axis sign ambiguity - prefer positive correlation
    angle_neg = calculate_joint_angle(q_rel, -jhat)
    n = min(len(angle_deg), len(gt_angles))
    corr_pos = np.corrcoef(angle_deg[:n], gt_angles[:n])[0, 1]
    corr_neg = np.corrcoef(angle_neg[:n], gt_angles[:n])[0, 1]

    # Pick the sign that gives better (more positive) correlation
    if corr_neg > corr_pos:
        jhat = -jhat
        angle_deg = angle_neg
        corr = corr_neg
    else:
        corr = corr_pos

    # Calculate RMSE
    rmse = np.sqrt(np.mean((gt_angles[:n] - angle_deg[:n])**2))

    return {
        'jhat': jhat,
        'angle_deg': angle_deg,
        'rmse': rmse,
        'corr': corr,
    }


def evaluate_axis_methods(
    subject_id,          # subject identifier (e.g., 'Subject08')
    joint,               # 'knee' or 'ankle'
    orientation='kf',    # 'kf', 'rnno', or 'both'
    methods=None,        # list of method names (default: all)
):
    """Evaluate multiple axis methods for a subject/joint, returns dict mapping (method, orientation) -> result."""
    cache_data = load_cache(subject_id, joint)
    if cache_data is None:
        raise ValueError(f"No cache for {subject_id}/{joint}. Run --precompute first.")

    if methods is None:
        methods = list(AXIS_METHODS.keys())

    orientations = ['kf', 'rnno'] if orientation == 'both' else [orientation]

    results = {}
    for ori in orientations:
        if ori == 'rnno' and 'q_rel_rnno' not in cache_data:
            print("Skipping RNNO (not in cache)")
            continue

        for method in methods:
            try:
                result = evaluate_axis_method(method, cache_data, joint, ori)
                key = (method, ori)
                results[key] = result
                print(f"{method} ({ori}): RMSE={result['rmse']:.2f}°, corr={result['corr']:.3f}")
            except Exception as e:
                print(f"{method} ({ori}): ERROR - {e}")

    return results


# =============================================================================
# Output
# =============================================================================

def print_results_table(all_results, joint):
    """Print formatted results table.

    Args:
        all_results: dict mapping subject_id -> results from evaluate_axis_methods
        joint: Joint name for display
    """
    # Collect all (method, orientation) keys
    all_keys = set()
    for results in all_results.values():
        all_keys.update(results.keys())
    all_keys = sorted(all_keys)

    if not all_keys:
        print("No results to display.")
        return

    # Build DataFrame rows
    rows = []
    for subject_id in sorted(all_results.keys()):
        results = all_results[subject_id]
        row = {'subject': subject_id}
        for key in all_keys:
            method, ori = key
            col_name = f"{method}_{ori}"
            if key in results:
                row[col_name] = results[key]['rmse']
            else:
                row[col_name] = np.nan
        rows.append(row)

    # Add mean row
    mean_row = {'subject': 'MEAN'}
    for key in all_keys:
        method, ori = key
        col_name = f"{method}_{ori}"
        vals = [r[col_name] for r in rows if not np.isnan(r.get(col_name, np.nan))]
        mean_row[col_name] = np.mean(vals) if vals else np.nan  # type: ignore[assignment]
    rows.append(mean_row)

    df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print(f"RMSE Summary - {joint.capitalize()} Joint (degrees)")
    print("="*80)
    print(df.to_string(index=False, float_format='%.2f'))

    return df


def save_results_csv(df, joint, output_dir='results'):
    """Save results DataFrame to CSV."""
    Path(output_dir).mkdir(exist_ok=True)
    csv_path = Path(output_dir) / f'axis_evaluation_{joint}.csv'
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\nResults saved to {csv_path}")


def plot_comparison(all_results, joint, subject_id=None, output_dir='plots'):
    """Plot time series comparison (optional)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    Path(output_dir).mkdir(exist_ok=True)

    # If no specific subject, use first one
    if subject_id is None:
        subject_id = sorted(all_results.keys())[0]

    results = all_results.get(subject_id, {})
    if not results:
        print(f"No results for {subject_id}")
        return

    # Load cache for ground truth
    cache_data = load_cache(subject_id, joint)
    if cache_data is None:
        return

    gt_angles = cache_data['gt_angles']
    fs = float(cache_data['fs'])
    time = np.arange(len(gt_angles)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time series plot
    ax1 = axes[0]
    ax1.plot(time, gt_angles, 'k-', label='Ground Truth', linewidth=1.5)

    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(results)))
    for (method, ori), color in zip(sorted(results.keys()), colors):
        angle = results[(method, ori)]['angle_deg']
        n = min(len(angle), len(time))
        ax1.plot(time[:n], angle[:n], label=f'{method} ({ori})', alpha=0.7)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (deg)')
    ax1.set_title(f'{joint.capitalize()} Joint Angle - {subject_id}')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # RMSE bar plot
    ax2 = axes[1]
    labels = [f'{m}\n({o})' for m, o in sorted(results.keys())]
    rmses = [results[k]['rmse'] for k in sorted(results.keys())]
    bars = ax2.bar(range(len(labels)), rmses, color=colors[:len(labels)])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('RMSE (deg)')
    ax2.set_title('RMSE Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{rmse:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    plot_path = Path(output_dir) / f'axis_evaluation_{joint}_{subject_id}.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(
        description='Evaluate joint axis estimation methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--joint', type=str, default='knee', choices=['knee', 'ankle'],
                        help='Joint to evaluate (default: knee)')
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID or "all" (default: Subject08)')
    parser.add_argument('--orientation', type=str, default='kf',
                        choices=['kf', 'rnno', 'both'],
                        help='Orientation source (default: kf)')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods (default: all)')
    parser.add_argument('--precompute', action='store_true',
                        help='Pre-compute and cache orientations')
    parser.add_argument('--no-rnno', action='store_true',
                        help='Skip RNNO computation in precompute')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available axis methods')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--no-csv', action='store_true',
                        help='Skip CSV output')

    args = parser.parse_args()

    # List methods
    if args.list_methods:
        print("\nAvailable axis estimation methods:")
        print("-" * 60)
        for name, info in AXIS_METHODS.items():
            flags = []
            if info['requires_gt']:
                flags.append('requires_gt')
            if info['requires_imu']:
                flags.append('requires_imu')
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {name}{flags_str}")
            print(f"    {info['func'].__doc__.strip().split(chr(10))[0]}")
        return

    # Determine subjects
    subjects = VALID_SUBJECTS if args.subject == 'all' else [args.subject]

    # Parse methods
    methods = args.methods.split(',') if args.methods else None

    # Pre-compute mode
    if args.precompute:
        for subject_id in subjects:
            try:
                precompute_orientations(subject_id, args.joint, include_rnno=not args.no_rnno)
            except Exception as e:
                print(f"Error pre-computing {subject_id}: {e}")
        return

    # Evaluation mode
    all_results = {}
    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"Evaluating: {subject_id} - {args.joint}")
        print(f"{'='*60}")
        try:
            results = evaluate_axis_methods(
                subject_id, args.joint,
                orientation=args.orientation,
                methods=methods
            )
            all_results[subject_id] = results
        except Exception as e:
            print(f"Error evaluating {subject_id}: {e}")

    # Output
    if all_results:
        df = print_results_table(all_results, args.joint)

        if df is not None and not args.no_csv:
            save_results_csv(df, args.joint)

        if args.plot:
            for subject_id in all_results:
                plot_comparison(all_results, args.joint, subject_id)


if __name__ == '__main__':
    main()
