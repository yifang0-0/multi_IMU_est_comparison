"""Export kf_gframe results to CSV for all subjects and joints."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from run_estimation import prepare_data
from methods.kf_gframe import run_kf_gframe_olsson

VALID_SUBJECTS = ['Subject01', 'Subject02', 'Subject03', 'Subject04',
                  'Subject07', 'Subject08', 'Subject11']
JOINTS = ['knee', 'ankle']


def export_single(subject_id, joint, output_dir):
    """Export kf_gframe results for one subject/joint."""
    data = prepare_data(joint, subject_id)

    angle_deg, r1, r2, jhat, q_rel = run_kf_gframe_olsson(
        data['acc_prox'], data['gyr_prox'],
        data['acc_dist'], data['gyr_dist'],
        data['fs']
    )

    # Truncate to common length
    n = min(len(angle_deg), len(data['gt']), len(data['acc_prox']))

    # Time-series dataframe
    time = np.arange(n) / data['fs']
    df = pd.DataFrame({
        'time': time,
        'q_w': q_rel[:n, 0],
        'q_x': q_rel[:n, 1],
        'q_y': q_rel[:n, 2],
        'q_z': q_rel[:n, 3],
        'angle_deg': angle_deg[:n],
        'gt_deg': data['gt'][:n],
        'acc_prox_x': data['acc_prox'][:n, 0],
        'acc_prox_y': data['acc_prox'][:n, 1],
        'acc_prox_z': data['acc_prox'][:n, 2],
        'gyr_prox_x': data['gyr_prox'][:n, 0],
        'gyr_prox_y': data['gyr_prox'][:n, 1],
        'gyr_prox_z': data['gyr_prox'][:n, 2],
        'acc_dist_x': data['acc_dist'][:n, 0],
        'acc_dist_y': data['acc_dist'][:n, 1],
        'acc_dist_z': data['acc_dist'][:n, 2],
        'gyr_dist_x': data['gyr_dist'][:n, 0],
        'gyr_dist_y': data['gyr_dist'][:n, 1],
        'gyr_dist_z': data['gyr_dist'][:n, 2],
    })

    # Metadata
    meta = pd.DataFrame([
        ('temporal_shift_samples', data['alignment_offset']),
        ('temporal_shift_seconds', data['alignment_offset'] / data['fs']),
        ('fs', data['fs']),
        ('jhat_x', jhat[0]),
        ('jhat_y', jhat[1]),
        ('jhat_z', jhat[2]),
        ('r1_x', r1[0]),
        ('r1_y', r1[1]),
        ('r1_z', r1[2]),
        ('r2_x', r2[0]),
        ('r2_y', r2[1]),
        ('r2_z', r2[2]),
    ], columns=['key', 'value'])

    # Write
    out = Path(output_dir)
    base = f"{subject_id}_{joint}_kf_gframe"
    df.to_csv(out / f"{base}.csv", index=False)
    meta.to_csv(out / f"{base}_meta.csv", index=False)

    return len(df)


def export_all(output_dir='exports'):
    """Export all subjects and joints."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    tasks = [(s, j) for s in VALID_SUBJECTS for j in JOINTS]

    for subject, joint in tqdm(tasks, desc="Exporting"):
        try:
            n = export_single(subject, joint, output_dir)
            tqdm.write(f"  {subject} {joint}: {n} samples")
        except Exception as e:
            tqdm.write(f"  {subject} {joint}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export kf_gframe results to CSV')
    parser.add_argument('--output-dir', default='exports')
    args = parser.parse_args()

    export_all(args.output_dir)
