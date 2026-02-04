"""Exhaustive global optimization of joint axis to check for local optima."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from methods.shared import calculate_joint_angle


def spherical_to_cart(theta, phi):
    """Convert spherical coordinates to Cartesian unit vector."""
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])


def create_objective(q_rel, gt):
    """Create objective function for axis optimization."""
    def objective(params):
        jhat = spherical_to_cart(*params)
        angle_est = calculate_joint_angle(q_rel, jhat)
        return np.sqrt(np.mean((gt - angle_est)**2))
    return objective


def current_4x8_grid(q_rel, gt):
    """Current implementation: 4x8 grid with L-BFGS-B."""
    objective = create_objective(q_rel, gt)
    init_points = [(theta, phi)
                   for theta in np.linspace(0.01, np.pi - 0.01, 4)
                   for phi in np.linspace(-np.pi, np.pi, 8, endpoint=False)]

    best = min(
        (minimize(objective, init, method='L-BFGS-B',
                  bounds=[(0, np.pi), (-np.pi, np.pi)])
         for init in init_points),
        key=lambda r: r.fun
    )
    return spherical_to_cart(*best.x), best.fun


def differential_evolution_search(q_rel, gt):
    """SciPy differential evolution - fast global optimizer."""
    objective = create_objective(q_rel, gt)
    bounds = [(0, np.pi), (-np.pi, np.pi)]
    result = differential_evolution(objective, bounds, maxiter=200, popsize=15,
                                    seed=42, polish=True, tol=0.001)
    return spherical_to_cart(*result.x), result.fun


def dual_annealing_search(q_rel, gt):
    """Dual annealing - simulated annealing variant."""
    objective = create_objective(q_rel, gt)
    bounds = [(0, np.pi), (-np.pi, np.pi)]
    result = dual_annealing(objective, bounds, maxiter=500, seed=42)
    return spherical_to_cart(*result.x), result.fun


def multistart_lbfgs(q_rel, gt, n_starts=50):
    """Random multistart with L-BFGS-B."""
    objective = create_objective(q_rel, gt)
    rng = np.random.default_rng(42)
    bounds = [(0, np.pi), (-np.pi, np.pi)]

    best_rmse, best_params = np.inf, None
    for _ in range(n_starts):
        x0 = [rng.uniform(0, np.pi), rng.uniform(-np.pi, np.pi)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        if result.fun < best_rmse:
            best_rmse, best_params = result.fun, result.x

    return spherical_to_cart(*best_params), best_rmse


def fix_axis_sign(axis, q_rel, gt):
    """Pick sign with better correlation to ground truth."""
    angle_pos = calculate_joint_angle(q_rel, axis)
    angle_neg = calculate_joint_angle(q_rel, -axis)
    if np.corrcoef(angle_neg, gt)[0, 1] > np.corrcoef(angle_pos, gt)[0, 1]:
        return -axis
    return axis


def main():
    results_dir = Path(__file__).parent.parent / 'results'

    print("=" * 70)
    print("JOINT AXIS OPTIMIZATION - Local Optima Investigation")
    print("=" * 70)

    for joint in ['knee', 'ankle']:
        print(f"\n{'='*60}")
        print(f"{joint.upper()} JOINT")
        print('='*60)

        for method in ['kf', 'rnno']:
            qrel_path = results_dir / f'qrel_{method}_{joint}.npy'
            gt_path = results_dir / f'gt_{joint}.npy'

            if not qrel_path.exists():
                print(f"\n  {method.upper()}: Run compare_joint_axes.py first")
                continue

            q_rel = np.load(qrel_path)
            gt = np.load(gt_path)

            print(f"\n  {method.upper()} q_rel ({len(q_rel)} samples)")
            print("  " + "-"*50)

            # Run optimization strategies
            print("  Running 4x8 grid (current)...", end=" ", flush=True)
            axis_4x8, rmse_4x8 = current_4x8_grid(q_rel, gt)
            axis_4x8 = fix_axis_sign(axis_4x8, q_rel, gt)
            print(f"RMSE={rmse_4x8:.4f}")

            print("  Running differential evolution...", end=" ", flush=True)
            axis_de, rmse_de = differential_evolution_search(q_rel, gt)
            axis_de = fix_axis_sign(axis_de, q_rel, gt)
            print(f"RMSE={rmse_de:.4f}")

            print("  Running dual annealing...", end=" ", flush=True)
            axis_da, rmse_da = dual_annealing_search(q_rel, gt)
            axis_da = fix_axis_sign(axis_da, q_rel, gt)
            print(f"RMSE={rmse_da:.4f}")

            print("  Running multistart L-BFGS (50x)...", end=" ", flush=True)
            axis_ms, rmse_ms = multistart_lbfgs(q_rel, gt, n_starts=50)
            axis_ms = fix_axis_sign(axis_ms, q_rel, gt)
            print(f"RMSE={rmse_ms:.4f}")

            # Report results
            results = [
                ('4x8 Grid (current)', rmse_4x8, axis_4x8),
                ('Differential Evolution', rmse_de, axis_de),
                ('Dual Annealing', rmse_da, axis_da),
                ('Multistart L-BFGS 50x', rmse_ms, axis_ms),
            ]

            print(f"\n  {'Method':<25} {'RMSE':<10} Axis")
            print("  " + "-"*60)
            for name, rmse, axis in results:
                print(f"  {name:<25} {rmse:<10.4f} [{axis[0]:7.4f}, {axis[1]:7.4f}, {axis[2]:7.4f}]")

            # Check convergence
            all_axes = [r[2] for r in results]
            max_diff = max(
                np.degrees(np.arccos(np.clip(abs(np.dot(ax1, ax2)), 0, 1)))
                for i, ax1 in enumerate(all_axes) for ax2 in all_axes[i+1:]
            )
            best_rmse = min(r[1] for r in results)
            rmse_spread = max(r[1] for r in results) - best_rmse

            print(f"\n  Max axis diff: {max_diff:.2f}° | RMSE spread: {rmse_spread:.4f}°")
            if max_diff < 1.0:
                print("  → All methods converge to same optimum ✓")
            elif max_diff < 5.0:
                print("  → Minor variations, likely numerical")
            else:
                print("  → SIGNIFICANT DIFFERENCES: Local optima issue!")


if __name__ == '__main__':
    main()
