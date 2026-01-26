import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import qmt
import xml.etree.ElementTree as ET
import argparse
import os

def load_imu_data(file_path):
    """Load raw IMU data from .txt file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('PacketCounter'):
            header_line = i
            data_start = i + 1
            break

    if header_line is None:
        raise ValueError(f"Could not find header in {file_path}")

    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=data_start,
        names=lines[header_line].strip().split('\t')
    )
    return df

def load_mot(file_path):
    """Load OpenSim motion (.mot) file."""
    header_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            header_lines += 1
            if line.strip().startswith('time'):
                break
    df = pd.read_csv(file_path, sep='\t', skiprows=header_lines-1)
    df.columns = df.columns.str.strip()
    return df

def get_sensor_mappings(xml_path):
    """Parse XML to get sensor mappings {body_part: sensor_id}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    mappings = {}
    sensors = root.find('.//ExperimentalSensors')
    if sensors is not None:
        for sensor in sensors.findall('ExperimentalSensor'):
            name = sensor.get('name')
            model_name = sensor.find('name_in_model').text
            mappings[model_name] = name
            
    return mappings

def estimate_orientations(acc, gyr, fs):
    """Estimate orientation using VQF without magnetometer."""
    params = {'Ts': 1.0/fs}
    q = qmt.oriEstVQF(gyr, acc, mag=None, params=params)
    if isinstance(q, tuple):
        q = q[0]
    return q

def calculate_ankle_angle_with_axis(q_tibia, q_calcn, jhat_tibia, jhat_calcn):
    """Calculate ankle angle using estimated joint axes via quaternion projection."""
    q_rel = qmt.qmult(qmt.qinv(q_tibia), q_calcn)
    q_twist = qmt.quatProject(q_rel, jhat_tibia)['projQuat']
    angle_mag = qmt.quatAngle(q_twist)
    twist_axis = qmt.quatAxis(q_twist)
    signs = np.sign(np.sum(twist_axis * jhat_tibia, axis=1))
    return np.degrees(angle_mag * signs)

def find_best_shift_and_axis(est_euler, gt_signal, max_shift_samples=10000):
    """Find best time shift and axis alignment using cross-correlation."""
    best_corr = 0
    best_shift = 0
    best_axis = 0
    best_sign = 1

    n = min(len(est_euler), len(gt_signal))

    if est_euler.ndim == 1:
        est_euler = est_euler.reshape(-1, 1)
        
    for axis in range(est_euler.shape[1]):
        sig1 = est_euler[:n, axis]
        sig2 = gt_signal[:n]
        
        lags = np.arange(-max_shift_samples, max_shift_samples + 1)
        
        if np.std(sig1) == 0 or np.std(sig2) == 0 or np.isnan(sig1).any() or np.isnan(sig2).any():
             corrs = np.zeros(len(lags))
        else:
             sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-6)
             sig2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-6)
             
             corrs = []
             for lag in lags:
                 if lag < 0:
                     c = np.corrcoef(sig1[-lag:], sig2[:lag])[0, 1]
                 elif lag > 0:
                     c = np.corrcoef(sig1[:-lag], sig2[lag:])[0, 1]
                 else:
                     c = np.corrcoef(sig1, sig2)[0, 1]
                 corrs.append(c)
             
             corrs = np.array(corrs)
        max_idx = np.argmax(np.abs(corrs))
        max_corr = corrs[max_idx]
        shift = lags[max_idx]
        
        print(f"Testing Axis {axis}: Best Shift found at {shift} samples (Corr: {max_corr:.4f})")

        if abs(max_corr) > abs(best_corr):
            best_corr = max_corr
            best_shift = shift
            best_axis = axis
            best_sign = np.sign(max_corr)
            
# --- ADD PRINT HERE TO SEE THE FINAL DECISION ---
    print(f"\n>>> FINAL SYNC DECISION:")
    print(f"Selected Axis: {best_axis}")
    print(f"Final Shift: {best_shift} samples")
    print(f"Signal Multiplier (Sign): {best_sign}")
    print(f"Maximum Correlation: {best_corr:.4f}\n")

    return best_shift, best_axis, best_sign, best_corr

def align_signals(est_euler, gt_signal, shift, axis, sign):
    """Align signals based on calculated shift."""
    if est_euler.ndim == 1:
        est_aligned = est_euler * sign
    else:
        est_aligned = est_euler[:, axis] * sign
    
    if shift > 0:
        common_len = min(len(est_aligned), len(gt_signal) - shift)
        est_final = est_aligned[:common_len]
        gt_final = gt_signal[shift:shift+common_len]
    elif shift < 0:
        s = -shift
        common_len = min(len(est_aligned) - s, len(gt_signal))
        est_final = est_aligned[s:s+common_len]
        gt_final = gt_signal[:common_len]
    else:
        common_len = min(len(est_aligned), len(gt_signal))
        est_final = est_aligned[:common_len]
        gt_final = gt_signal[:common_len]
        
    return est_final, gt_final

def plot_time_series_error(errors_dict, save_path='plots/time_series_error_ankle.png', no_plot=False, num_entries=3):
    """Create a time series plot of errors for the first N entries."""
    if not errors_dict:
        print("No errors to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    labels = list(errors_dict.keys())[:num_entries]

    for i, label in enumerate(labels):
        errors = errors_dict[label]
        time_samples = np.arange(len(errors))
        ax.plot(time_samples, errors, label=label, color=colors[i % len(colors)],
                alpha=0.8, linewidth=1.0)

    ax.set_title('Ankle Angle Estimation Error Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sample', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Error (degrees)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 1000)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    stats_lines = []
    for label in labels:
        rmse = np.sqrt(np.mean(np.array(errors_dict[label])**2))
        stats_lines.append(f"{label}: RMSE={rmse:.2f}°")
    stats_text = '\n'.join(stats_lines)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series error plot saved to {save_path}")

    if not no_plot:
        plt.show()
    plt.close()

def plot_error_comparison(errors_dict, save_path='plots/error_comparison_ankle.png', no_plot=False):
    """Create a boxplot comparison of errors from different algorithms."""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = list(errors_dict.keys())
    data = [errors_dict[label] for label in labels]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False,
                       widths=0.6, medianprops=dict(color='black', linewidth=2),
                       boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

    for patch, color in zip(bplot['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title('Ankle Angle Estimation Error Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Absolute Error (degrees)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    means = [np.mean(d) for d in data]
    rmses = [np.sqrt(np.mean(np.array(d)**2)) for d in data]
    medians = [np.median(d) for d in data]

    stats_text = '\n'.join([
        f"{labels[i]}: RMSE={rmses[i]:.2f}°  |  Mean={means[i]:.2f}°  |  Median={medians[i]:.2f}°"
        for i in range(len(labels))
    ])

    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error comparison plot saved to {save_path}")

    if not no_plot:
        plt.show()
    plt.close()

def load_opensense_results(subject_path):
    """Load pre-calculated results from OpenSense algorithms."""
    algos = ['xsens', 'madgwick', 'mahony']
    results = {}
    
    for algo in algos:
        path = subject_path / 'IMU' / algo / 'IKResults' / 'IKWithErrorsUniformWeights' / 'walking_IK.mot'
        if not path.exists():
             path = subject_path / 'IMU' / algo / 'IKResults' / 'IKWithErrorsExtremeLowFeetWeights' / 'walking_IK.mot'
        
        if path.exists():
            df = load_mot(path)
            if 'ankle_angle_r' in df.columns:
                results[algo] = df['ankle_angle_r'].values
            
    return results

def prepare_data():
    subject_path = Path('Subject08/walking')
    imu_path = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    mapping_path = subject_path / 'IMU' / 'myIMUMappings_walking.xml'
    fs = 100.0
    
    mappings = get_sensor_mappings(mapping_path)
    tibia_r_id = mappings.get('tibia_r_imu')
    calcn_r_id = mappings.get('calcn_r_imu')
    
    if not tibia_r_id or not calcn_r_id:
        raise ValueError("Could not find sensor IDs")
        
    tibia_file = list(imu_path.glob(f"*{tibia_r_id}.txt"))[0]
    calcn_file = list(imu_path.glob(f"*{calcn_r_id}.txt"))[0]
    
    tibia_df = load_imu_data(tibia_file)
    calcn_df = load_imu_data(calcn_file)
    
    q_tibia = estimate_orientations(
        tibia_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values,
        tibia_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values,
        fs
    )
    q_calcn = estimate_orientations(
        calcn_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values,
        calcn_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values,
        fs
    )
    
    gt_df = load_mot(subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot')
    ankle_gt = gt_df['ankle_angle_r'].values
    
    acc_tibia = tibia_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_tibia = tibia_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    acc_calcn = calcn_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_calcn = calcn_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    
    return {
        'q_tibia': q_tibia,
        'q_calcn': q_calcn,
        'acc_tibia': acc_tibia,
        'gyr_tibia': gyr_tibia,
        'acc_calcn': acc_calcn,
        'gyr_calcn': gyr_calcn,
        'fs': fs,
        'ankle_gt': ankle_gt,
        'subject_path': subject_path
    }

def estimate_joint_axes(data):
    """Estimate joint axes using Olsson method."""
    jhat_tibia, jhat_calcn = qmt.jointAxisEstHingeOlsson(
        data['acc_tibia'], data['acc_calcn'],
        data['gyr_tibia'], data['gyr_calcn']
    )
    return jhat_tibia.flatten(), jhat_calcn.flatten()

def run_vqf_olsson(data, args, errors_dict, jhat_tibia, jhat_calcn):
    print("\n=== VQF+Olsson Joint Axis Estimation ===")
    q_tibia = data['q_tibia']
    q_calcn = data['q_calcn']
    ankle_gt = data['ankle_gt']
    fs = data['fs']

    ankle_angle_olsson = calculate_ankle_angle_with_axis(q_tibia, q_calcn, jhat_tibia, jhat_calcn)
    ankle_angle_olsson_neg = calculate_ankle_angle_with_axis(q_tibia, q_calcn, -jhat_tibia, -jhat_calcn)

    corr_pos = np.corrcoef(ankle_angle_olsson[:min(len(ankle_angle_olsson), len(ankle_gt))],
                           ankle_gt[:min(len(ankle_angle_olsson), len(ankle_gt))])[0, 1]
    corr_neg = np.corrcoef(ankle_angle_olsson_neg[:min(len(ankle_angle_olsson_neg), len(ankle_gt))],
                           ankle_gt[:min(len(ankle_angle_olsson_neg), len(ankle_gt))])[0, 1]

    if abs(corr_neg) > abs(corr_pos):
        ankle_angle_olsson = ankle_angle_olsson_neg
        best_jhat_tibia = -jhat_tibia
        best_jhat_calcn = -jhat_calcn
    else:
        best_jhat_tibia = jhat_tibia
        best_jhat_calcn = jhat_calcn

    shift, axis, sign, corr = find_best_shift_and_axis(ankle_angle_olsson, ankle_gt)
    est_final, gt_final = align_signals(ankle_angle_olsson, ankle_gt, shift, axis, sign)

    rmse = np.sqrt(np.mean((gt_final - est_final)**2))
    print(f"VQF+Olsson Method - RMSE: {rmse:.2f} degrees")

    errors_dict['vqf+olsson'] = np.abs(gt_final - est_final)
    return best_jhat_tibia, best_jhat_calcn, est_final, gt_final

def run_opensense(data, args, errors_dict):
    print("\n=== OpenSense Comparison ===")
    subject_path = data['subject_path']
    ankle_gt = data['ankle_gt']
    
    opensense_results = load_opensense_results(subject_path)
    
    for algo_name, algo_data in opensense_results.items():
        n = min(len(ankle_gt), len(algo_data))
        gt_segment = ankle_gt[:n]
        algo_segment = algo_data[:n]
        
        error = np.abs(gt_segment - algo_segment)
        errors_dict[algo_name.capitalize()] = error
        
        rmse_algo = np.sqrt(np.mean(error**2))
        print(f"{algo_name.capitalize()} - RMSE: {rmse_algo:.2f} degrees")

def calculate_hinge_angle(gyr1, acc1, gyr2, acc2, Ts):
    """Calculate joint angle for a hinge joint using VQF, Olsson axes, and heading correction."""
    q_imu1 = qmt.oriEstVQF(gyr1, acc1, params={'Ts': Ts})
    q_imu2 = qmt.oriEstVQF(gyr2, acc2, params={'Ts': Ts})

    j1, j2 = qmt.jointAxisEstHingeOlsson(acc1, acc2, gyr1, gyr2)
    j1 = j1.squeeze()
    j2 = j2.squeeze()

    print(f"Estimated Axis 1: {j1}")
    print(f"Estimated Axis 2: {j2}")

    q_align1 = qmt.quatFrom2Axes(z=j1, x=acc1[0], exactAxis='z')
    q_align2 = qmt.quatFrom2Axes(z=j2, x=acc2[0], exactAxis='z')

    q_seg1 = qmt.qmult(q_imu1, q_align1)
    q_seg2 = qmt.qmult(q_imu2, q_align2)

    t = qmt.timeVec(N=q_seg1.shape[0], Ts=Ts)

    out = qmt.headingCorrection(
        gyr1=gyr1,
        gyr2=gyr2,
        quat1=q_seg1,
        quat2=q_seg2,
        t=t,
        joint=[0, 0, 1],
        jointInfo={},
        estSettings={'constraint': 'euler_1d'}
    )

    q_seg2_corrected = out[0]

    q_rel = qmt.qrel(q_seg1, q_seg2_corrected)
    angles = qmt.eulerAngles(q_rel, axes='zyx')
    joint_angle = np.unwrap(angles[:, 0])
    return joint_angle

def run_vqf_olsson_heading_correction(data, args, errors_dict):
    print("\n=== VQF+Olsson+Heading_Correction ===")
    acc_tibia = data['acc_tibia']
    gyr_tibia = data['gyr_tibia']
    acc_calcn = data['acc_calcn']
    gyr_calcn = data['gyr_calcn']
    fs = data['fs']
    ankle_gt = data['ankle_gt']

    Ts = 1.0 / fs
    
    angle_rad = calculate_hinge_angle(gyr_tibia, acc_tibia, gyr_calcn, acc_calcn, Ts)
    angle_deg = np.degrees(angle_rad)
    
    shift, axis, sign, corr = find_best_shift_and_axis(angle_deg, ankle_gt)
    est_final, gt_final = align_signals(angle_deg, ankle_gt, shift, axis, sign)
    
    rmse = np.sqrt(np.mean((gt_final - est_final)**2))
    print(f"VQF+Olsson+Heading_Correction - RMSE: {rmse:.2f} degrees")

    errors_dict['vqf+olsson+heading_correction'] = np.abs(gt_final - est_final)


# need to change the data structure to what the OPT can use
# 


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current Working Directory set to: {os.getcwd()}")
    parser = argparse.ArgumentParser(description='Run Ankle Estimation')
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting')
    parser.add_argument('--method', type=str, default='all', choices=['vqf_olsson', 'heading_correction', 'opensense', 'vqf_olsson_heading_correction', 'all'], help='Method to run')
    args = parser.parse_args()
    
    Path('plots').mkdir(exist_ok=True)

    data = prepare_data()
    errors_dict = {}

    jhat_tibia = None
    jhat_calcn = None
    
    if args.method in ['vqf_olsson', 'heading_correction', 'all']:
        jhat_tibia, jhat_calcn = estimate_joint_axes(data)

    if args.method in ['vqf_olsson', 'all']:
        best_tibia, best_calcn, est_final_o, gt_final_o = run_vqf_olsson(data, args, errors_dict, jhat_tibia, jhat_calcn)
        jhat_tibia = best_tibia
        jhat_calcn = best_calcn

    if args.method in ['vqf_olsson_heading_correction', 'all']:
        run_vqf_olsson_heading_correction(data, args, errors_dict)

    if args.method in ['opensense', 'all']:
        run_opensense(data, args, errors_dict)


    if errors_dict:
        plot_time_series_error(errors_dict, no_plot=args.no_plot, num_entries=3)
        plot_error_comparison(errors_dict, no_plot=args.no_plot)

if __name__ == "__main__":
    main()

