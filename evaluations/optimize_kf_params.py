"""Optimize MEKF-acc or MAP-acc noise parameters via differential evolution."""
import sys
import os
import argparse
import time
import multiprocessing as mp
import numpy as np
import qmt
from scipy.optimize import differential_evolution

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_estimation import prepare_data
from constants import VALID_SUBJECTS
from dfjimu import estimate_lever_arms, run_mekf_cython, map_acc
from dfjimu.utils.common import preprocess_acc_at_center
from methods.axis import estimate_joint_axis
from methods.shared import calculate_joint_angle
from methods.kf_gframe import _Q_COV, _R_DIAG, _P_INIT_DIAG

JOINTS = ['knee', 'ankle']

# Module-level refs set by pool initializer for worker processes
_GLOBAL_DATASETS = None
_GLOBAL_OPT_PINIT = False
_GLOBAL_AXIS_METHOD = 'opensim'
_GLOBAL_ESTIMATOR = 'mekf'


def _init_worker(datasets, opt_pinit, axis_method, estimator='mekf'):
    """Pool initializer — sets globals in each worker process."""
    global _GLOBAL_DATASETS, _GLOBAL_OPT_PINIT, _GLOBAL_AXIS_METHOD, _GLOBAL_ESTIMATOR
    _GLOBAL_DATASETS = datasets
    _GLOBAL_OPT_PINIT = opt_pinit
    _GLOBAL_AXIS_METHOD = axis_method
    _GLOBAL_ESTIMATOR = estimator


def precompute_datasets():
    """Load all subject/joint combos and precompute lever arms + center-of-joint accelerations."""
    datasets = []
    for subject in VALID_SUBJECTS:
        for joint in JOINTS:
            print(f"Loading {subject} / {joint}...")
            data = prepare_data(joint, subject)
            gyr1, gyr2 = data['gyr_prox'], data['gyr_dist']
            acc1, acc2 = data['acc_prox'], data['acc_dist']
            fs = data['fs']

            r1, r2 = estimate_lever_arms(gyr1, gyr2, acc1, acc2, fs)
            C1 = preprocess_acc_at_center(gyr1, acc1, r1, fs)
            C2 = preprocess_acc_at_center(gyr2, acc2, r2, fs)

            datasets.append({
                'subject': subject,
                'joint': joint,
                'gyr1': gyr1.astype(np.float64),
                'gyr2': gyr2.astype(np.float64),
                'acc1': acc1.astype(np.float64),
                'acc2': acc2.astype(np.float64),
                'C1': C1.astype(np.float64),
                'C2': C2.astype(np.float64),
                'r1': r1,
                'r2': r2,
                'fs': fs,
                'gt': data['gt'],
            })
    return datasets


def evaluate_single(ds, Q_cov, R_diag, P_init_diag=1.0, axis_method='opensim'):
    """Run MEKF + axis estimation on one dataset, return RMSE in degrees."""
    q_init = np.array([1.0, 0, 0, 0], dtype=np.float64)
    Q_arr = np.ones(6, dtype=np.float64) * Q_cov

    q1, q2 = run_mekf_cython(
        ds['gyr1'], ds['gyr2'], ds['acc1'], ds['acc2'],
        ds['C1'], ds['C2'], ds['fs'], q_init, Q_arr, R_diag, P_init_diag,
    )

    q_rel = qmt.qmult(qmt.qinv(q1), q2)

    axis_kwargs = {'correct_sign': True}
    if axis_method == 'optimized':
        axis_kwargs.update(gt_angles=ds['gt'], calib_samples=3000)
    elif axis_method == 'opensim':
        axis_kwargs['joint'] = ds['joint']

    jhat = estimate_joint_axis(q_rel, axis_method=axis_method, **axis_kwargs)
    angle_deg = calculate_joint_angle(q_rel, jhat)

    n = min(len(angle_deg), len(ds['gt']))
    rmse = np.sqrt(np.mean((angle_deg[:n] - ds['gt'][:n]) ** 2))
    return rmse


def evaluate_single_map(ds, cov_w_scale, cov_i_scale, cov_lnk_scale, axis_method='opensim'):
    """Run MAP-acc + axis estimation on one dataset, return RMSE in degrees."""
    q1, q2 = map_acc(
        ds['gyr1'], ds['gyr2'], ds['acc1'], ds['acc2'],
        r1=ds['r1'], r2=ds['r2'], Fs=ds['fs'],
        q_init=np.array([1.0, 0, 0, 0], dtype=np.float64),
        cov_w=np.eye(6) * cov_w_scale,
        cov_i=np.eye(3) * cov_i_scale,
        cov_lnk=np.eye(3) * cov_lnk_scale,
    )

    q_rel = qmt.qmult(qmt.qinv(q1), q2)

    axis_kwargs = {'correct_sign': True}
    if axis_method == 'optimized':
        axis_kwargs.update(gt_angles=ds['gt'], calib_samples=3000)
    elif axis_method == 'opensim':
        axis_kwargs['joint'] = ds['joint']

    jhat = estimate_joint_axis(q_rel, axis_method=axis_method, **axis_kwargs)
    angle_deg = calculate_joint_angle(q_rel, jhat)

    n = min(len(angle_deg), len(ds['gt']))
    rmse = np.sqrt(np.mean((angle_deg[:n] - ds['gt'][:n]) ** 2))
    return rmse


def objective(log_params):
    """Mean RMSE across all datasets. Parameters in log10 space."""
    rmses = []
    for ds in _GLOBAL_DATASETS:
        try:
            if _GLOBAL_ESTIMATOR == 'map':
                cov_w = 10 ** log_params[0]
                cov_i = 10 ** log_params[1]
                cov_lnk = 10 ** log_params[2]
                rmses.append(evaluate_single_map(ds, cov_w, cov_i, cov_lnk, _GLOBAL_AXIS_METHOD))
            else:
                Q_cov = 10 ** log_params[0]
                R_diag = 10 ** log_params[1]
                P_init_diag = 10 ** log_params[2] if _GLOBAL_OPT_PINIT else 1.0
                rmses.append(evaluate_single(ds, Q_cov, R_diag, P_init_diag, _GLOBAL_AXIS_METHOD))
        except Exception:
            rmses.append(50.0)  # penalty

    return np.mean(rmses)


def compute_rmse_map(datasets, Q_cov, R_diag, P_init_diag=1.0, axis_method='opensim'):
    """Evaluate all MEKF datasets and return {(subject, joint): rmse} map."""
    return {
        (ds['subject'], ds['joint']): evaluate_single(ds, Q_cov, R_diag, P_init_diag, axis_method)
        for ds in datasets
    }


def compute_rmse_map_map(datasets, cov_w_scale, cov_i_scale, cov_lnk_scale, axis_method='opensim'):
    """Evaluate all MAP-acc datasets and return {(subject, joint): rmse} map."""
    return {
        (ds['subject'], ds['joint']): evaluate_single_map(ds, cov_w_scale, cov_i_scale, cov_lnk_scale, axis_method)
        for ds in datasets
    }


def print_rmse_table(rmse_map, label='', params_str=''):
    """Print subject x joint RMSE table from a precomputed rmse_map, return mean RMSE."""
    if label:
        print(f"\n=== {label} ===")
    if params_str:
        print(f"  {params_str}")

    subjects = sorted(set(s for s, _ in rmse_map))
    joints = sorted(set(j for _, j in rmse_map))

    # Header
    print(f"  {'Subject':<12}", end='')
    for j in joints:
        print(f"  {j:>8}", end='')
    print(f"  {'Mean':>8}")

    # Per-subject rows
    for subject in subjects:
        print(f"  {subject:<12}", end='')
        row = [rmse_map[(subject, j)] for j in joints if (subject, j) in rmse_map]
        for r in row:
            print(f"  {r:8.2f}", end='')
        print(f"  {np.mean(row):8.2f}")

    # Mean row
    print(f"  {'MEAN':<12}", end='')
    all_vals = list(rmse_map.values())
    for j in joints:
        col = [rmse_map[(s, j)] for s in subjects if (s, j) in rmse_map]
        print(f"  {np.mean(col):8.2f}", end='')
    print(f"  {np.mean(all_vals):8.2f}")

    return np.mean(all_vals)


def run_optimization(datasets, bounds, opt_pinit=False, axis_method='opensim',
                     maxiter=50, popsize=15, seed=42, workers=-1, estimator='mekf'):
    """Run differential evolution and return best parameters."""
    # Also set globals in the main process (used by callback's print)
    global _GLOBAL_DATASETS, _GLOBAL_OPT_PINIT, _GLOBAL_AXIS_METHOD, _GLOBAL_ESTIMATOR
    _GLOBAL_DATASETS = datasets
    _GLOBAL_OPT_PINIT = opt_pinit
    _GLOBAL_AXIS_METHOD = axis_method
    _GLOBAL_ESTIMATOR = estimator

    t0 = time.time()
    gen = [0]

    def callback(xk, convergence):
        gen[0] += 1
        if estimator == 'map':
            cw, ci, cl = 10 ** xk[0], 10 ** xk[1], 10 ** xk[2]
            print(f"  gen {gen[0]:3d} | conv={convergence:.4f} | "
                  f"cov_w={cw:.2e} cov_i={ci:.2e} cov_lnk={cl:.2e} | {time.time() - t0:.0f}s")
        else:
            Q, R = 10 ** xk[0], 10 ** xk[1]
            print(f"  gen {gen[0]:3d} | conv={convergence:.4f} | "
                  f"Q={Q:.2e} R={R:.2e} | {time.time() - t0:.0f}s")

    n_workers = workers if workers > 0 else mp.cpu_count()
    pool = mp.Pool(n_workers, initializer=_init_worker,
                   initargs=(datasets, opt_pinit, axis_method, estimator))

    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            tol=1e-4,
            workers=pool.map,
            updating='deferred',
            disp=True,
            callback=callback,
        )
    finally:
        pool.close()
        pool.join()

    params = [10 ** x for x in result.x]

    print(f"\nOptimization finished: {result.nfev} evaluations, success={result.success}")
    if estimator == 'map':
        print(f"  Best: cov_w={params[0]:.4e}, cov_i={params[1]:.4e}, cov_lnk={params[2]:.4e}")
    else:
        print(f"  Best: Q_cov={params[0]:.4e}, R_diag={params[1]:.4e}", end='')
        if opt_pinit:
            print(f", P_init_diag={params[2]:.4e}")
        else:
            print()
    print(f"  Mean RMSE: {result.fun:.4f}")

    return params, result


def _format_params_mekf(Q, R, P, axis_method):
    """Format MEKF parameter string."""
    return f"Q_cov={Q:.2e}, R_diag={R:.4f}, P_init_diag={P:.4f}, axis={axis_method}"


def _format_params_map(cov_w, cov_i, cov_lnk, axis_method):
    """Format MAP parameter string."""
    return f"cov_w={cov_w:.2e}, cov_i={cov_i:.2e}, cov_lnk={cov_lnk:.2e}, axis={axis_method}"


def main():
    """CLI entrypoint for KF parameter optimization."""
    parser = argparse.ArgumentParser(description='Optimize MEKF-acc or MAP-acc noise parameters')
    parser.add_argument('--estimator', type=str, default='mekf', choices=['mekf', 'map'],
                        help='Estimator to optimize (default: mekf)')
    parser.add_argument('--maxiter', type=int, default=50, help='DE max iterations (default: 50)')
    parser.add_argument('--popsize', type=int, default=15, help='DE population size (default: 15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--per-joint', action='store_true', help='Optimize per-joint separately')
    parser.add_argument('--opt-pinit', action='store_true', help='Also optimize P_init_diag (3rd param, MEKF only)')
    parser.add_argument('--axis-method', type=str, default='opensim',
                        choices=['opensim', 'optimized', 'olsson', 'pca_rotvec'],
                        help='Axis estimation method (default: opensim)')
    parser.add_argument('--baseline-only', action='store_true', help='Only print baseline RMSE table')
    parser.add_argument('--workers', type=int, default=-1, help='Parallel workers for DE (-1=all CPUs, default: -1)')
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")

    use_map = args.estimator == 'map'

    # MEKF baseline defaults
    BASELINE_Q = float(_Q_COV[0])
    BASELINE_R = float(_R_DIAG)
    BASELINE_P = float(_P_INIT_DIAG)

    # MAP baseline defaults
    from constants import COV_W_SCALE
    BASELINE_COV_W = COV_W_SCALE
    BASELINE_COV_I = COV_W_SCALE
    BASELINE_COV_LNK = _R_DIAG

    print("\nPreloading datasets...")
    datasets = precompute_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    axis_method = args.axis_method

    # Baseline
    if use_map:
        params_str = _format_params_map(BASELINE_COV_W, BASELINE_COV_I, BASELINE_COV_LNK, axis_method)
        baseline_map = compute_rmse_map_map(datasets, BASELINE_COV_W, BASELINE_COV_I, BASELINE_COV_LNK, axis_method)
    else:
        params_str = _format_params_mekf(BASELINE_Q, BASELINE_R, BASELINE_P, axis_method)
        baseline_map = compute_rmse_map(datasets, BASELINE_Q, BASELINE_R, BASELINE_P, axis_method)
    baseline_rmse = print_rmse_table(baseline_map, label='Baseline (current parameters)',
                                     params_str=params_str)

    if args.baseline_only:
        return

    # Bounds in log10 space
    if use_map:
        bounds = [(-5, 1), (-5, 1), (-5, 1)]  # cov_w, cov_i, cov_lnk
    elif args.opt_pinit:
        bounds = [(-5, 1), (-3, 3), (-2, 2)]  # Q_cov, R_diag, P_init_diag
    else:
        bounds = [(-5, 1), (-3, 3)]            # Q_cov, R_diag

    def _run_and_print(ds, label_suffix=''):
        params, _ = run_optimization(ds, bounds, args.opt_pinit, axis_method,
                                     args.maxiter, args.popsize, args.seed,
                                     args.workers, args.estimator)
        if use_map:
            opt_map = compute_rmse_map_map(ds, params[0], params[1], params[2], axis_method)
            ps = _format_params_map(params[0], params[1], params[2], axis_method)
        else:
            Q, R = params[0], params[1]
            P = params[2] if args.opt_pinit else 1.0
            opt_map = compute_rmse_map(ds, Q, R, P, axis_method)
            ps = _format_params_mekf(Q, R, P, axis_method)
        label = f'Optimized ({label_suffix})' if label_suffix else 'Optimized'
        opt_rmse = print_rmse_table(opt_map, label=label, params_str=ps)
        return params, opt_rmse

    if args.per_joint:
        for joint in JOINTS:
            print(f"\n{'='*60}")
            print(f"Optimizing for {joint} only")
            print(f"{'='*60}")
            joint_ds = [d for d in datasets if d['joint'] == joint]
            _run_and_print(joint_ds, label_suffix=joint)
    else:
        params, opt_rmse = _run_and_print(datasets)

        print("\n=== Summary ===")
        print(f"  Baseline mean RMSE: {baseline_rmse:.4f}")
        print(f"  Optimized mean RMSE: {opt_rmse:.4f}")
        print(f"  Delta: {opt_rmse - baseline_rmse:+.4f}")
        if use_map:
            print(f"\n  cov_w_scale = {params[0]:.6e}")
            print(f"  cov_i_scale = {params[1]:.6e}")
            print(f"  cov_lnk_scale = {params[2]:.6e}")
        else:
            print(f"\n  Q_cov = {params[0]:.6e}")
            print(f"  R_diag = {params[1]:.6e}")
            if args.opt_pinit:
                print(f"  P_init_diag = {params[2]:.6e}")
            print("\nTo update, edit methods/kf_gframe.py lines 11-13")


if __name__ == '__main__':
    main()
