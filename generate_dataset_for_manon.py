"""
Generate a test dataset by saving IMU signals, joint axes (Olsson), lever arms, and GT angles.

Input data is read from  dataset_temp_manon/<subject>/walking/  (same structure as run_estimation.py
expects under data/, but using the local dataset_temp_manon folder instead).

Saves per subject/joint into testset_manon/<subject>_<joint>.npz with keys:
  acc_prox  (N, 3)  proximal accelerometer [m/s^2]
  gyr_prox  (N, 3)  proximal gyroscope [rad/s]
  acc_dist  (N, 3)  distal accelerometer [m/s^2]
  gyr_dist  (N, 3)  distal gyroscope [rad/s]
  j1        (3,)    Olsson joint axis in proximal sensor frame
  j2        (3,)    Olsson joint axis in distal sensor frame
  r1_est    (3,)    lever arm proximal (from estimate_lever_arms)
  r2_est    (3,)    lever arm distal  (from estimate_lever_arms)
  gt        (N,)    ground truth joint angle [degrees]
  fs        ()      sampling frequency [Hz]

Usage:
    python generate_dataset.py --joint knee --subject Subject08
    python generate_dataset.py --joint ankle --subject all
    python generate_dataset.py --joint all --subject all
"""
import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Stub out optional packages not needed here (avoids ImportError from methods/__init__.py)
for _mod in ('imt', 'imt.methods', 'optuna'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from dfjimu import estimate_lever_arms
from methods.vqf_olsson import run_vqf_olsson
from methods.shared import load_mot
from utils import load_imu_data, get_sensor_mappings, get_aligned_time_range
from constants import VALID_SUBJECTS

# Data root — where subject folders live
DATA_ROOT = 'dataset_temp_manon'

JOINT_CONFIG = {
    'knee': {
        'proximal_sensor': 'femur_r_imu',
        'distal_sensor':   'tibia_r_imu',
        'gt_column':       'knee_angle_r',
    },
    'ankle': {
        'proximal_sensor': 'tibia_r_imu',
        'distal_sensor':   'calcn_r_imu',
        'gt_column':       'ankle_angle_r',
    },
}


def _load_subject_data(joint_name: str, subject_id: str):
    """Load and align IMU + GT data from dataset_temp_manon, mirroring prepare_data() logic."""
    cfg = JOINT_CONFIG[joint_name]
    subject_path = Path(DATA_ROOT) / subject_id / 'walking'
    imu_path = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    fs = 100.0

    mappings  = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    prox_id   = mappings.get(cfg['proximal_sensor'])
    dist_id   = mappings.get(cfg['distal_sensor'])
    if not prox_id or not dist_id:
        raise ValueError(f"Sensor IDs not found for {joint_name}")

    prox_df = load_imu_data(list(imu_path.glob(f"*{prox_id}.txt"))[0])
    dist_df = load_imu_data(list(imu_path.glob(f"*{dist_id}.txt"))[0])

    acc_prox = prox_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_prox = prox_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    acc_dist = dist_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_dist = dist_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

    gt_df = load_mot(subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot')
    gt    = gt_df[cfg['gt_column']].values

    time_range = get_aligned_time_range(subject_path, int(fs))
    imu_start  = time_range['imu_start']
    imu_end    = time_range['imu_end']
    offset     = time_range['offset']
    print(f"  Alignment offset: {offset} samples ({offset/fs:.2f} sec)")

    acc_prox = acc_prox[imu_start:imu_end]
    gyr_prox = gyr_prox[imu_start:imu_end]
    acc_dist = acc_dist[imu_start:imu_end]
    gyr_dist = gyr_dist[imu_start:imu_end]

    n = min(len(acc_prox), len(acc_dist), len(gt))
    acc_prox, gyr_prox = acc_prox[:n], gyr_prox[:n]
    acc_dist, gyr_dist = acc_dist[:n], gyr_dist[:n]
    gt = gt[:n]
    print(f"  Aligned length: {n} samples ({n/fs:.1f} sec)")

    return acc_prox, gyr_prox, acc_dist, gyr_dist, fs, gt


def generate_for_subject(joint: str, subject_id: str, output_dir: Path) -> None:
    print(f"\n--- {subject_id}  {joint} ---")

    try:
        acc_prox, gyr_prox, acc_dist, gyr_dist, fs, gt = _load_subject_data(joint, subject_id)
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    # Joint axes via VQF + Olsson
    _, j1, j2, _, _, _ = run_vqf_olsson(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)

    # Lever arms (same call as kf_gframe's estimate_lever_arms step)
    r1, r2 = estimate_lever_arms(gyr_prox, gyr_dist, acc_prox, acc_dist, fs)

    out_path = output_dir / f"{subject_id}_{joint}.npz"
    np.savez(
        out_path,
        acc_prox=acc_prox,
        gyr_prox=gyr_prox,
        acc_dist=acc_dist,
        gyr_dist=gyr_dist,
        j1=j1,
        j2=j2,
        r1_est=r1,
        r2_est=r2,
        gt=gt,
        fs=np.float64(fs),
    )
    print(f"  Saved {out_path}  (N={len(gt)}, fs={fs} Hz)")
    print(f"  j1={j1}")
    print(f"  j2={j2}")
    print(f"  r1_est={r1}")
    print(f"  r2_est={r2}")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Generate test dataset as .npz files',
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--joint',   default='knee', choices=['knee', 'ankle', 'all'])
    parser.add_argument('--subject', default='Subject08',
                        help='Subject ID or "all" (default: Subject08)')
    parser.add_argument('--output',  default='testset_manon',
                        help='Output directory for .npz files (default: testset_manon)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    joints   = ['knee', 'ankle'] if args.joint  == 'all' else [args.joint]
    subjects = VALID_SUBJECTS    if args.subject == 'all' else [args.subject]

    for joint in joints:
        for subject_id in subjects:
            generate_for_subject(joint, subject_id, output_dir)

    print(f"\nDone. Files written to '{output_dir}/'")


if __name__ == '__main__':
    main()
