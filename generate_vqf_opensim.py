"""
Generate orientation .sto files for OpenSim IMU Inverse Kinematics.

This script runs VQF orientation estimation on all IMU sensors and outputs
.sto files compatible with OpenSim's IMU IK tool. Optionally runs OpenSim
IMU IK to generate joint angle .mot files.

Supports using existing Madgwick orientations for validation against
precomputed IK results.

Usage:
    python generate_vqf_opensim.py                           # VQF for Subject08
    python generate_vqf_opensim.py --subject all             # VQF for all subjects
    python generate_vqf_opensim.py --run-ik                  # Generate and run IK
    python generate_vqf_opensim.py --run-ik --subject all    # Parallel IK for all subjects
    python generate_vqf_opensim.py --method madgwick --run-ik  # Use existing Madgwick
    python generate_vqf_opensim.py --method madgwick --validate  # Compare to existing
"""
import numpy as np
import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import qmt

from utils import load_imu_data, get_sensor_mappings, write_orientations_sto, read_orientations_sto
from methods.shared import load_mot

VALID_SUBJECTS = ['Subject02', 'Subject03', 'Subject04', 'Subject08']

# Standard sensor names in OpenSim order
SENSOR_NAMES = ['pelvis_imu', 'femur_r_imu', 'femur_l_imu',
                'tibia_r_imu', 'tibia_l_imu', 'calcn_r_imu']


def generate_vqf_orientations(subject_id):
    """Generate VQF orientations for all sensors and write to .sto file."""
    subject_path = Path(f'data/{subject_id}/walking')
    mappings = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    fs = 100.0

    quaternions = {}
    time = None

    for sensor_name in SENSOR_NAMES:
        sensor_id = mappings.get(sensor_name)
        if not sensor_id:
            print(f"  {sensor_name}: not mapped, skipping")
            continue

        # Find IMU file
        sensor_id_clean = sensor_id.lstrip('_')
        imu_files = list(imu_dir.glob(f"*{sensor_id_clean}.txt"))
        if not imu_files:
            print(f"  {sensor_name}: file not found for {sensor_id}, skipping")
            continue

        # Load IMU data
        imu_df = load_imu_data(imu_files[0])
        acc = imu_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyr = imu_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
        mag = imu_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values

        # Run VQF orientation estimation with magnetometer for Earth-fixed heading
        q = qmt.oriEstVQF(gyr, acc, mag=mag, params={'Ts': 1.0/fs})
        quaternions[sensor_name] = q
        print(f"  {sensor_name}: {len(q)} samples")

        if time is None:
            time = np.arange(len(q)) / fs

    if not quaternions:
        raise ValueError(f"No valid sensors found for {subject_id}")

    # Truncate all to minimum length (some sensors may have slightly different counts)
    min_len = min(len(q) for q in quaternions.values())
    time = time[:min_len]
    quaternions = {name: q[:min_len] for name, q in quaternions.items()}

    # Write to .sto file
    output_path = subject_path / 'IMU' / 'vqf' / 'walking_orientations.sto'
    write_orientations_sto(output_path, time, quaternions, SENSOR_NAMES, int(fs))
    print(f"  Output: {output_path} ({min_len} samples)")

    return output_path


def run_opensim_ik(subject_id, orientations_sto, method='vqf', low_feet_weights=False):
    """Run OpenSim IMU Inverse Kinematics on orientation data."""
    import opensim as osim

    subject_path = Path(f'data/{subject_id}/walking')
    model_path = subject_path / 'IMU' / 'madgwick' / 'model_Rajagopal2015_calibrated.osim'

    # Use separate output directory for madgwick validation to avoid overwriting originals
    if method == 'madgwick':
        output_dir = subject_path / 'IMU' / 'madgwick_reproduced' / 'IKResults'
    else:
        output_dir = subject_path / 'IMU' / method / 'IKResults'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and configure IMU IK tool
    ik_tool = osim.IMUInverseKinematicsTool()
    ik_tool.set_model_file(str(model_path))
    ik_tool.set_orientations_file(str(orientations_sto))
    ik_tool.set_results_directory(str(output_dir))

    # Sensor-to-OpenSim rotation: -90° around X axis (from existing setup XML)
    ik_tool.set_sensor_to_opensim_rotations(osim.Vec3(-1.5707963, 0, 0))

    # Set accuracy (matches existing setup XML)
    ik_tool.set_accuracy(1e-6)

    # Set orientation weights (low feet weights improves ankle angle estimation)
    if low_feet_weights:
        weights = {
            'pelvis_imu': 1.0, 'femur_r_imu': 1.0, 'femur_l_imu': 1.0,
            'tibia_r_imu': 0.5, 'tibia_l_imu': 0.5,
            'calcn_r_imu': 0.01, 'calcn_l_imu': 0.01,
        }
        weight_set = ik_tool.upd_orientation_weights()
        for sensor, weight in weights.items():
            w = osim.OrientationWeight(sensor, weight)
            weight_set.cloneAndAppend(w)

    # Run IK
    ik_tool.run()

    # Rename output files to match madgwick convention
    (output_dir / 'ik_walking_orientations.mot').rename(output_dir / 'walking_IK.mot')
    (output_dir / 'ik_walking_orientations_orientationErrors.sto').rename(
        output_dir / 'walking_orientationErrors.sto'
    )

    return output_dir / 'walking_IK.mot'


def validate_results(new_mot, existing_mot):
    """Compare two .mot files, report RMSE for key angles."""
    new_df = load_mot(new_mot)
    existing_df = load_mot(existing_mot)

    # Align by time if needed (should be identical for same orientations)
    min_len = min(len(new_df), len(existing_df))
    new_df = new_df.iloc[:min_len]
    existing_df = existing_df.iloc[:min_len]

    print(f"  Comparing {min_len} samples:")
    for col in ['knee_angle_r', 'ankle_angle_r']:
        if col in new_df.columns and col in existing_df.columns:
            rmse = np.sqrt(np.mean((new_df[col] - existing_df[col])**2))
            max_diff = np.max(np.abs(new_df[col] - existing_df[col]))
            print(f"    {col}: RMSE = {rmse:.6f}°, max diff = {max_diff:.6f}°")
        else:
            print(f"    {col}: not found in one of the files")


def process_subject(args_tuple):
    """Process a single subject (worker function for multiprocessing)."""
    subj, method, run_ik, validate, low_feet_weights = args_tuple
    results = {'subject': subj, 'status': 'success', 'messages': []}

    try:
        subject_path = Path(f'data/{subj}/walking')

        if method == 'vqf':
            sto_path = generate_vqf_orientations(subj)
            results['messages'].append(f"Generated orientations: {sto_path}")
        else:
            # Use existing Madgwick orientations - convert to compatible format
            madgwick_sto = subject_path / 'IMU' / 'madgwick' / 'walking_orientations.sto'
            if not madgwick_sto.exists():
                results['status'] = 'error'
                results['messages'].append(f"Error: {madgwick_sto} not found")
                return results

            # Convert to compatible format
            time, quaternions, data_rate = read_orientations_sto(madgwick_sto)
            sto_path = subject_path / 'IMU' / 'madgwick_reproduced' / 'walking_orientations.sto'
            write_orientations_sto(sto_path, time, quaternions, list(quaternions.keys()), data_rate)
            results['messages'].append(f"Converted orientations: {sto_path}")

        if run_ik:
            mot_path = run_opensim_ik(subj, sto_path, method=method, low_feet_weights=low_feet_weights)
            results['messages'].append(f"IK output: {mot_path}")
            results['mot_path'] = str(mot_path)

        results['sto_path'] = str(sto_path)

    except Exception as e:
        results['status'] = 'error'
        results['messages'].append(f"Error: {e}")

    return results


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='Generate orientation .sto files for OpenSim IK',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID or "all" for all valid subjects')
    parser.add_argument('--method', choices=['vqf', 'madgwick'], default='vqf',
                        help='Orientation method: vqf (generate) or madgwick (use existing)')
    parser.add_argument('--run-ik', action='store_true',
                        help='Run OpenSim IMU IK after generating orientations')
    parser.add_argument('--validate', action='store_true',
                        help='Compare IK results against existing (madgwick only)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially instead of parallel (for debugging)')
    parser.add_argument('--low-feet-weights', action='store_true',
                        help='Use low orientation weights for foot sensors (0.01)')
    args = parser.parse_args()

    subjects = VALID_SUBJECTS if args.subject == 'all' else [args.subject]

    if args.method == 'vqf':
        print(f"Generating VQF orientation .sto files for: {', '.join(subjects)}")
    else:
        print(f"Using existing Madgwick orientations for: {', '.join(subjects)}")

    # Use parallel processing for multiple subjects with IK
    use_parallel = len(subjects) > 1 and args.run_ik and not args.sequential

    if use_parallel:
        n_workers = min(len(subjects), cpu_count())
        print(f"Running IK in parallel with {n_workers} workers...\n")

        work_items = [(subj, args.method, args.run_ik, args.validate, args.low_feet_weights) for subj in subjects]

        with Pool(n_workers) as pool:
            results = pool.map(process_subject, work_items)

        # Print results
        for result in results:
            print(f"\n{result['subject']}:")
            for msg in result['messages']:
                print(f"  {msg}")

            # Run validation after parallel IK (validation is fast, do sequentially)
            if args.validate and args.method == 'madgwick' and result['status'] == 'success':
                subject_path = Path(f"data/{result['subject']}/walking")
                new_mot = subject_path / 'IMU' / 'madgwick_reproduced' / 'IKResults' / 'walking_IK.mot'
                existing_mot = subject_path / 'IMU' / 'madgwick' / 'IKResults' / 'IKWithErrorsUniformWeights' / 'walking_IK.mot'
                if new_mot.exists() and existing_mot.exists():
                    print(f"  Validating against: {existing_mot}")
                    validate_results(new_mot, existing_mot)

    else:
        # Sequential processing
        for subj in subjects:
            print(f"\n{subj}:")
            try:
                subject_path = Path(f'data/{subj}/walking')

                if args.method == 'vqf':
                    sto_path = generate_vqf_orientations(subj)
                else:
                    # Use existing Madgwick orientations - convert to compatible format
                    madgwick_sto = subject_path / 'IMU' / 'madgwick' / 'walking_orientations.sto'
                    if not madgwick_sto.exists():
                        print(f"  Error: {madgwick_sto} not found")
                        continue
                    print(f"  Reading: {madgwick_sto}")

                    # Convert to compatible format
                    time, quaternions, data_rate = read_orientations_sto(madgwick_sto)
                    sto_path = subject_path / 'IMU' / 'madgwick_reproduced' / 'walking_orientations.sto'
                    write_orientations_sto(sto_path, time, quaternions, list(quaternions.keys()), data_rate)
                    print(f"  Converted to: {sto_path}")

                if args.run_ik:
                    print(f"  Running OpenSim IMU IK...")
                    mot_path = run_opensim_ik(subj, sto_path, method=args.method, low_feet_weights=args.low_feet_weights)
                    print(f"  IK output: {mot_path}")

                    # Auto-validate for madgwick
                    if args.method == 'madgwick':
                        existing_mot = subject_path / 'IMU' / 'madgwick' / 'IKResults' / 'IKWithErrorsUniformWeights' / 'walking_IK.mot'
                        if existing_mot.exists():
                            print(f"  Validating against: {existing_mot}")
                            validate_results(mot_path, existing_mot)

                if args.validate and args.method == 'madgwick':
                    new_mot = subject_path / 'IMU' / 'madgwick_reproduced' / 'IKResults' / 'walking_IK.mot'
                    existing_mot = subject_path / 'IMU' / 'madgwick' / 'IKResults' / 'IKWithErrorsUniformWeights' / 'walking_IK.mot'
                    if new_mot.exists() and existing_mot.exists():
                        print(f"  Validating against: {existing_mot}")
                        validate_results(new_mot, existing_mot)
                    else:
                        print(f"  Cannot validate: run with --run-ik first")

            except Exception as e:
                print(f"  Error: {e}")

    if not args.run_ik and args.method == 'vqf':
        print("\nNext steps:")
        print("  Run with --run-ik to execute OpenSim IMU IK")


if __name__ == "__main__":
    main()
