"""Compare optimized joint axes between KF_gframe and RNNO methods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from run_estimation import prepare_data
from methods import run_kf_gframe_optimized, run_rnno_optimized


def angular_difference(v1, v2):
    """Angle between two unit vectors in degrees (handles sign ambiguity)."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(abs(dot)))  # abs handles sign ambiguity


def save_qrel_data(joint, q_rel_kf, q_rel_rnno, gt, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'results'
    """Save q_rel arrays for exhaustive optimization analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{output_dir}/qrel_kf_{joint}.npy', q_rel_kf)
    np.save(f'{output_dir}/qrel_rnno_{joint}.npy', q_rel_rnno)
    np.save(f'{output_dir}/gt_{joint}.npy', gt)
    print(f"  Saved q_rel arrays to {output_dir}/qrel_*_{joint}.npy")


def main():
    for joint in ['knee', 'ankle']:
        print(f"\n{'='*50}")
        print(f"{joint.upper()} JOINT (Subject08)")
        print('='*50)

        data = prepare_data(joint, 'Subject08')

        # Run KF_gframe with optimized axis (returns q_rel)
        _, r1, r2, jhat_kf, q_rel_kf = run_kf_gframe_optimized(
            data['acc_prox'], data['gyr_prox'],
            data['acc_dist'], data['gyr_dist'],
            data['fs'], data['gt']
        )

        # Run RNNO with optimized axis (returns q_rel)
        _, _, _, jhat_rnno, q_rel_rnno = run_rnno_optimized(
            data['acc_prox'], data['gyr_prox'],
            data['acc_dist'], data['gyr_dist'],
            data['fs'], data['gt']
        )

        print(f"  KF_gframe axis: [{jhat_kf[0]:7.4f}, {jhat_kf[1]:7.4f}, {jhat_kf[2]:7.4f}]")
        print(f"  RNNO axis:      [{jhat_rnno[0]:7.4f}, {jhat_rnno[1]:7.4f}, {jhat_rnno[2]:7.4f}]")
        print(f"  Angular diff:   {angular_difference(jhat_kf, jhat_rnno):.2f}°")

        # Check if axes are aligned or opposite
        dot = np.dot(jhat_kf, jhat_rnno)
        if dot < 0:
            print(f"  Note: Axes point in opposite directions (dot={dot:.3f})")

        # Save q_rel arrays for exhaustive optimization analysis
        save_qrel_data(joint, q_rel_kf, q_rel_rnno, data['gt'])


if __name__ == '__main__':
    main()
