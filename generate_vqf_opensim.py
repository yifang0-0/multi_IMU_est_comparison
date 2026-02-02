"""
Generate orientation .sto files for OpenSim IMU Inverse Kinematics.

This script runs VQF orientation estimation on all IMU sensors and outputs
.sto files compatible with OpenSim's IMU IK tool. Optionally runs OpenSim
IMU IK to generate joint angle .mot files.

For VQF method, follows the complete OpenSense calibration workflow:
1. Generate VQF orientations
2. Run IMUPlacer calibration with VQF orientations
3. Apply heading correction
4. Run IK with VQF-calibrated model and corrected orientations

Supports using existing Madgwick orientations for validation against
precomputed IK results.

Usage:
    python generate_vqf_opensim.py                           # VQF for Subject08
    python generate_vqf_opensim.py --subject all             # VQF for all subjects
    python generate_vqf_opensim.py --run-ik                  # Generate and run IK
    python generate_vqf_opensim.py --run-ik --subject all    # Parallel IK for all subjects
    python generate_vqf_opensim.py --method madgwick --run-ik  # Use existing Madgwick
    python generate_vqf_opensim.py --method madgwick --validate  # Compare to existing
    python generate_vqf_opensim.py --run-ik --both-weights  # Run both weight configs
"""
import numpy as np
import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import qmt

from utils import load_imu_data, get_sensor_mappings, write_orientations_sto, read_orientations_sto, get_aligned_time_range
from methods.shared import load_mot
from constants import VALID_SUBJECTS

# Standard sensor names in OpenSim order
SENSOR_NAMES = ['pelvis_imu', 'femur_r_imu', 'femur_l_imu',
                'tibia_r_imu', 'tibia_l_imu', 'calcn_r_imu']


def generate_vqf_orientations(subject_id, align_to_mocap=True):
    """Generate VQF orientations for all sensors and write to .sto file.

    Args:
        subject_id: Subject identifier (e.g., 'Subject03')
        align_to_mocap: If True, trim IMU data to match ground truth duration.
            This prevents heading drift during pre-recording periods.
    """
    subject_path = Path(f'data/{subject_id}/walking')
    mappings = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    fs = 100.0

    # Get aligned time range (start and end truncation to match GT duration)
    trim_start = 0
    trim_end = None  # None means no end truncation
    if align_to_mocap:
        time_range = get_aligned_time_range(subject_path, fs)
        trim_start = time_range['imu_start']
        trim_end = time_range['imu_end']
        if trim_start > 0 or trim_end is not None:
            duration = (trim_end - trim_start) / fs if trim_end else 'unknown'
            print(f"  Aligning to GT: samples [{trim_start}:{trim_end}] ({duration:.1f}s)")

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

        # Trim to aligned time range (both start and end)
        if trim_start > 0 or trim_end is not None:
            acc = acc[trim_start:trim_end]
            gyr = gyr[trim_start:trim_end]
            mag = mag[trim_start:trim_end]

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


def run_imu_placer(subject_id, orientations_sto, posed_model_path):
    """Run IMUPlacer to create calibrated model with VQF orientations."""
    import opensim as osim

    subject_path = Path(f'data/{subject_id}/walking')
    output_dir = subject_path / 'IMU' / 'vqf'
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_model_path = output_dir / 'model_Rajagopal2015_calibrated.osim'

    imu_placer = osim.IMUPlacer()
    imu_placer.set_model_file(str(posed_model_path))
    imu_placer.set_orientation_file_for_calibration(str(orientations_sto))
    imu_placer.set_base_imu_label('pelvis_imu')
    imu_placer.set_base_heading_axis('z')
    imu_placer.set_sensor_to_opensim_rotations(osim.Vec3(-np.pi/2, 0, 0))

    imu_placer.run(False)  # visualize=False

    calibrated_model = imu_placer.getCalibratedModel()
    calibrated_model.printToXML(str(calibrated_model_path))

    print(f"  IMUPlacer output: {calibrated_model_path}")
    return calibrated_model_path


def apply_heading_correction(subject_id, orientations_sto, posed_model_path):
    """Apply heading correction to orientation data."""
    import opensim as osim

    subject_path = Path(f'data/{subject_id}/walking')
    marker_ik_path = subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot'
    output_sto = subject_path / 'IMU' / 'vqf' / 'walking_orientations_hc.sto'

    # Load model and get pelvis rotation from marker IK
    model = osim.Model(str(posed_model_path))
    state = model.initSystem()
    model.realizePosition(state)

    marker_motion = osim.TimeSeriesTable(str(marker_ik_path))
    col_idx = marker_motion.getColumnIndex('pelvis_rotation')
    pelvis_rotation = marker_motion.getRowAtIndex(0)[col_idx]

    # Load orientations and apply sensor-to-opensim rotation for heading computation
    osense = osim.OpenSenseUtilities()
    oTable = osim.TimeSeriesTableQuaternion(str(orientations_sto))

    R_sensor = osim.Rotation()
    R_sensor.setRotationFromAngleAboutX(-np.pi/2)
    osense.rotateOrientationTable(oTable, R_sensor)

    # Compute heading correction
    heading_axis = osim.CoordinateDirection(osim.CoordinateAxis(2), 1)  # +Z
    correction_vec = osim.OpenSenseUtilities.computeHeadingCorrection(
        model, state, oTable, 'pelvis_imu', heading_axis)
    computed_correction = correction_vec.get(1) * 180 / np.pi  # to degrees

    # Apply full correction (computed - pelvis rotation from marker IK)
    angular_correction = computed_correction - pelvis_rotation

    # Reload original orientations and apply Z-rotation
    oTable_final = osim.TimeSeriesTableQuaternion(str(orientations_sto))
    R_heading = osim.Rotation()
    R_heading.setRotationFromAngleAboutZ(np.radians(angular_correction))
    osense.rotateOrientationTable(oTable_final, R_heading)

    osim.STOFileAdapterQuaternion.write(oTable_final, str(output_sto))

    print(f"  Heading correction: {angular_correction:.2f}° (computed={computed_correction:.2f}°, pelvis={pelvis_rotation:.2f}°)")
    print(f"  Output: {output_sto}")
    return output_sto


def run_opensim_ik(subject_id, orientations_sto, method='vqf', low_feet_weights=False,
                   model_path=None):
    """Run OpenSim IMU Inverse Kinematics on orientation data."""
    import opensim as osim

    subject_path = Path(f'data/{subject_id}/walking')
    if model_path is None:
        model_path = subject_path / 'IMU' / 'madgwick' / 'model_Rajagopal2015_calibrated.osim'

    # Match directory structure of other methods: IKResults/{weighting}/walking_IK.mot
    weighting_dir = 'IKWithErrorsExtremeLowFeetWeights' if low_feet_weights else 'IKWithErrorsUniformWeights'
    if method == 'madgwick':
        output_dir = subject_path / 'IMU' / 'madgwick_reproduced' / 'IKResults' / weighting_dir
    else:
        output_dir = subject_path / 'IMU' / method / 'IKResults' / weighting_dir
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
    # Output name is based on orientations file: ik_{basename}.mot
    ori_basename = Path(orientations_sto).stem
    (output_dir / f'ik_{ori_basename}.mot').rename(output_dir / 'walking_IK.mot')
    (output_dir / f'ik_{ori_basename}_orientationErrors.sto').rename(
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
    subj, method, run_ik, _validate, low_feet_weights = args_tuple
    results = {'subject': subj, 'status': 'success', 'messages': []}

    try:
        subject_path = Path(f'data/{subj}/walking')

        if method == 'vqf':
            # Step 1: Generate VQF orientations
            sto_path = generate_vqf_orientations(subj)
            results['messages'].append(f"Generated orientations: {sto_path}")

            # Step 2: Use existing posed model (from marker IK, same across methods)
            posed_model = subject_path / 'IMU' / 'madgwick' / 'model_Rajagopal2015_posed.osim'

            # Step 3: Run IMUPlacer calibration with VQF orientations
            calibrated_model = run_imu_placer(subj, sto_path, posed_model)
            results['messages'].append(f"Calibrated model: {calibrated_model}")

            # Step 4: Apply heading correction
            corrected_sto = apply_heading_correction(subj, sto_path, posed_model)
            results['messages'].append(f"Heading-corrected orientations: {corrected_sto}")

            # Use corrected orientations and calibrated model for IK
            ik_orientations = corrected_sto
            ik_model = calibrated_model
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
            ik_orientations = sto_path
            ik_model = None  # Use default

        if run_ik:
            mot_path = run_opensim_ik(subj, ik_orientations, method=method,
                                       low_feet_weights=low_feet_weights,
                                       model_path=ik_model)
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
    parser.add_argument('--both-weights', action='store_true',
                        help='Run IK with both uniform and low feet weights in parallel')
    args = parser.parse_args()

    subjects = VALID_SUBJECTS if args.subject == 'all' else [args.subject]

    if args.method == 'vqf':
        print(f"Generating VQF orientation .sto files for: {', '.join(subjects)}")
    else:
        print(f"Using existing Madgwick orientations for: {', '.join(subjects)}")

    # Determine weight configurations to run
    weight_configs = [False, True] if args.both_weights else [args.low_feet_weights]

    # Use parallel processing for multiple subjects with IK or both weight configs
    use_parallel = (len(subjects) > 1 or args.both_weights) and args.run_ik and not args.sequential

    if use_parallel:
        work_items = [(subj, args.method, args.run_ik, args.validate, weights)
                      for subj in subjects for weights in weight_configs]
        n_workers = min(len(work_items), cpu_count())
        print(f"Running IK in parallel with {n_workers} workers ({len(work_items)} tasks)...\n")

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
                    # Step 1: Generate VQF orientations
                    sto_path = generate_vqf_orientations(subj)

                    # Step 2: Use existing posed model (from marker IK, same across methods)
                    posed_model = subject_path / 'IMU' / 'madgwick' / 'model_Rajagopal2015_posed.osim'

                    # Step 3: Run IMUPlacer calibration with VQF orientations
                    calibrated_model = run_imu_placer(subj, sto_path, posed_model)

                    # Step 4: Apply heading correction
                    corrected_sto = apply_heading_correction(subj, sto_path, posed_model)

                    # Use corrected orientations and calibrated model for IK
                    ik_orientations = corrected_sto
                    ik_model = calibrated_model
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
                    ik_orientations = sto_path
                    ik_model = None  # Use default

                if args.run_ik:
                    for low_feet_weights in weight_configs:
                        weight_label = "low feet weights" if low_feet_weights else "uniform weights"
                        print(f"  Running OpenSim IMU IK ({weight_label})...")
                        mot_path = run_opensim_ik(subj, ik_orientations, method=args.method,
                                                   low_feet_weights=low_feet_weights,
                                                   model_path=ik_model)
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
                        print("  Cannot validate: run with --run-ik first")

            except Exception as e:
                print(f"  Error: {e}")

    if not args.run_ik and args.method == 'vqf':
        print("\nNext steps:")
        print("  Run with --run-ik to execute OpenSim IMU IK")


if __name__ == "__main__":
    main()
