import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import qmt
import xml.etree.ElementTree as ET
import argparse
import os
from KF_Gframe import process_orientation_KF_Gframe


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

def calculate_knee_angle_with_axis(q_femur, q_tibia, jhat_femur, jhat_tibia):
    """Calculate knee angle using estimated joint axes via quaternion projection."""
    q_rel = qmt.qmult(qmt.qinv(q_femur), q_tibia)
    q_twist = qmt.quatProject(q_rel, jhat_femur)['projQuat']
    angle_mag = qmt.quatAngle(q_twist)
    twist_axis = qmt.quatAxis(q_twist)
    signs = np.sign(np.sum(twist_axis * jhat_femur, axis=1))
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
        
        if abs(max_corr) > abs(best_corr):
            best_corr = max_corr
            best_shift = shift
            best_axis = axis
            best_sign = np.sign(max_corr)
            
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

def plot_time_series_error(errors_dict, save_path='plots/time_series_error.png', no_plot=False, num_entries=4):
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

    ax.set_title('Knee Angle Estimation Error Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sample', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Error (degrees)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 1000)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add RMSE stats in legend area
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

def plot_error_comparison(errors_dict, save_path='plots/error_comparison.png', no_plot=False):
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

    ax.set_title('Knee Angle Estimation Error Distribution', fontsize=16, fontweight='bold', pad=20)
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
            if 'knee_angle_r' in df.columns:
                results[algo] = df['knee_angle_r'].values
            
    return results

def prepare_data():
    subject_path = Path('Subject08/walking')
    imu_path = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    mapping_path = subject_path / 'IMU' / 'myIMUMappings_walking.xml'
    fs = 100.0
    
    mappings = get_sensor_mappings(mapping_path)
    femur_r_id = mappings.get('femur_r_imu')
    tibia_r_id = mappings.get('tibia_r_imu')
    
    if not femur_r_id or not tibia_r_id:
        raise ValueError("Could not find sensor IDs")
        
    femur_file = list(imu_path.glob(f"*{femur_r_id}.txt"))[0]
    tibia_file = list(imu_path.glob(f"*{tibia_r_id}.txt"))[0]
    
    femur_df = load_imu_data(femur_file)
    tibia_df = load_imu_data(tibia_file)
    
    q_femur = estimate_orientations(
        femur_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values,
        femur_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values,
        fs
    )
    q_tibia = estimate_orientations(
        tibia_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values,
        tibia_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values,
        fs
    )
    
    gt_df = load_mot(subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot')
    knee_gt = gt_df['knee_angle_r'].values
    
    acc_femur = femur_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_femur = femur_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    acc_tibia = tibia_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_tibia = tibia_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    
    return {
        'q_femur': q_femur,
        'q_tibia': q_tibia,
        'acc_femur': acc_femur,
        'gyr_femur': gyr_femur,
        'acc_tibia': acc_tibia,
        'gyr_tibia': gyr_tibia,
        'fs': fs,
        'knee_gt': knee_gt,
        'subject_path': subject_path
    }

def estimate_joint_axes(data):
    """Estimate joint axes using Olsson method."""
    jhat_femur, jhat_tibia = qmt.jointAxisEstHingeOlsson(
        data['acc_femur'], data['acc_tibia'],
        data['gyr_femur'], data['gyr_tibia']
    )
    return jhat_femur.flatten(), jhat_tibia.flatten()

def run_vqf_olsson(data, args, errors_dict, jhat_femur, jhat_tibia):
    print("\n=== VQF+Olsson Joint Axis Estimation ===")
    q_femur = data['q_femur']
    q_tibia = data['q_tibia']
    knee_gt = data['knee_gt']
    fs = data['fs']
    
    knee_angle_olsson = calculate_knee_angle_with_axis(q_femur, q_tibia, jhat_femur, jhat_tibia)
    knee_angle_olsson_neg = calculate_knee_angle_with_axis(q_femur, q_tibia, -jhat_femur, -jhat_tibia)
    
    corr_pos = np.corrcoef(knee_angle_olsson[:min(len(knee_angle_olsson), len(knee_gt))], 
                           knee_gt[:min(len(knee_angle_olsson), len(knee_gt))])[0, 1]
    corr_neg = np.corrcoef(knee_angle_olsson_neg[:min(len(knee_angle_olsson_neg), len(knee_gt))], 
                           knee_gt[:min(len(knee_angle_olsson_neg), len(knee_gt))])[0, 1]
    
    if abs(corr_neg) > abs(corr_pos):
        knee_angle_olsson = knee_angle_olsson_neg
        best_jhat_femur = -jhat_femur
        best_jhat_tibia = -jhat_tibia
    else:
        best_jhat_femur = jhat_femur
        best_jhat_tibia = jhat_tibia
    
    shift, axis, sign, corr = find_best_shift_and_axis(knee_angle_olsson, knee_gt)
    est_final, gt_final = align_signals(knee_angle_olsson, knee_gt, shift, axis, sign)
    
    rmse = np.sqrt(np.mean((gt_final - est_final)**2))
    print(f"VQF+Olsson Method - RMSE: {rmse:.2f} degrees")

    errors_dict['vqf+olsson'] = np.abs(gt_final - est_final)
    return best_jhat_femur, best_jhat_tibia, est_final, gt_final

def run_opensense(data, args, errors_dict):
    print("\n=== OpenSense Comparison ===")
    subject_path = data['subject_path']
    knee_gt = data['knee_gt']
    
    opensense_results = load_opensense_results(subject_path)
    
    for algo_name, algo_data in opensense_results.items():
        n = min(len(knee_gt), len(algo_data))
        gt_segment = knee_gt[:n]
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
    acc_femur = data['acc_femur']
    gyr_femur = data['gyr_femur']
    acc_tibia = data['acc_tibia']
    gyr_tibia = data['gyr_tibia']
    fs = data['fs']
    knee_gt = data['knee_gt']

    Ts = 1.0 / fs
    
    angle_rad = calculate_hinge_angle(gyr_femur, acc_femur, gyr_tibia, acc_tibia, Ts)
    angle_deg = np.degrees(angle_rad)
    
    shift, axis, sign, corr = find_best_shift_and_axis(angle_deg, knee_gt)
    est_final, gt_final = align_signals(angle_deg, knee_gt, shift, axis, sign)
    
    rmse = np.sqrt(np.mean((gt_final - est_final)**2))
    print(f"VQF+Olsson+Heading_Correction - RMSE: {rmse:.2f} degrees")

    errors_dict['vqf+olsson+heading_correction'] = np.abs(gt_final - est_final)

# def run_kf_gframe_comparison(data_orig, fs, errors_dict):
#     Ts = 1.0 / fs
#     N = len(data_orig['acc_femur'])

#     class KFParams:
#         def __init__(self):
#             self.N = N
#             self.T = Ts
#             self.Fs = fs
#             self.q1 = np.array([1, 0, 0, 0]) 
#             self.cov_w = np.eye(6) * 1e-4   
#             self.cov_lnk = np.eye(3) * 1e-2  
#             self.if_cheat_dw = False
#             self.if_cheat_w = False
#             self.if_cheat_a = False
#             self.run_dynamic_update = True
#             self.run_measurement_update = True
            
#         def __getitem__(self, key): 
#             return getattr(self, key)

#     import GlobalParams
#     GlobalParams.params = KFParams() 

#     kf_input_data = {
#         'gyr_1': data_orig['gyr_femur'].T,
#         'gyr_2': data_orig['gyr_tibia'].T,
#         'acc_1': data_orig['acc_femur'].T,
#         'acc_2': data_orig['acc_tibia'].T,
#         'r1': np.array([0, 0, 0.2]).reshape(3,1), 
#         'r2': np.array([0, 0, -0.2]).reshape(3,1)
#     }

#     q1_all, q2_all, P_list = process_orientation_KF_Gframe(kf_input_data)
    
#     q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)
    
#     angles = qmt.eulerAngles(q_rel, axes='zyx')
#     joint_angle = np.unwrap(angles[:, 0])
    
#     return joint_angle

import numpy as np
import qmt
import GlobalParams as params
from KF_Gframe import process_orientation_KF_Gframe

def run_kf_gframe_with_auto_r(data, args, errors_dict):
    acc1 = data['acc_femur'].T
    gyr1 = data['gyr_femur'].T
    acc2 = data['acc_tibia'].T
    gyr2 = data['gyr_tibia'].T
    fs = data['fs']
    knee_gt = data['knee_gt']
    Ts = 1.0 / fs
    print("all data shapes: acc1,gyr1,acc2,gyr2", acc1.shape, gyr1.shape, acc2.shape, gyr2.shape)
    print("fs:", fs )
    print("")
    N = acc1.shape[1]
    

    def estimate_r(a1, g1, a2, g2, sampling_fs, iterations=25, step=0.7):
        def get_dgyr(y, f):
            dy = np.zeros_like(y)
            dy[:, 2:-2] = (y[:, :-4] - 8 * y[:, 1:-3] + 8 * y[:, 3:-1] - y[:, 4:]) * (f / 12)
            return dy
        
        def get_K(g, dg):
            num = g.shape[1]
            K_mat = np.zeros((3, 3, num))
            for i in range(num):
                w, alpha = g[:, i], dg[:, i]
                Sw = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
                Sa = np.array([[0, -alpha[2], alpha[1]], [alpha[2], 0, -alpha[0]], [-alpha[1], alpha[0], 0]])
                K_mat[:, :, i] = (Sw @ Sw) + Sa
            return K_mat

        dg1, dg2 = get_dgyr(g1, sampling_fs), get_dgyr(g2, sampling_fs)
        K1, K2 = get_K(g1, dg1), get_K(g2, dg2)
        
        x = 0.1 * np.ones(6)
        num = g1.shape[1]
        for _ in range(iterations):
            e1 = a1 - np.array([K1[:,:,i] @ x[0:3] for i in range(num)]).T
            e2 = a2 - np.array([K2[:,:,i] @ x[3:6] for i in range(num)]).T
            
            n1, n2 = np.linalg.norm(e1, axis=0), np.linalg.norm(e2, axis=0)
            eps = n1 - n2
            
            J = np.zeros((num, 6))
            for i in range(num):
                J[i, 0:3] = -(K1[:,:,i].T @ e1[:,i]) / (n1[i] + 1e-9)
                J[i, 3:6] = (K2[:,:,i].T @ e2[:,i]) / (n2[i] + 1e-9)
            
            G = J.T @ eps
            H = J.T @ J
            try:
                x -= step * np.linalg.solve(H + 1e-8 * np.eye(6), G)
            except np.linalg.LinAlgError:
                break
        return x[0:3], x[3:6]

    # r1_est, r2_est = estimate_r(acc1, gyr1, acc2, gyr2, fs)
    r1_est = np.array([-0.1222504, 0.01730777, -0.00477925])
    r2_est = np.array([-0.03597717, -0.01554343, -0.0232674])
    print(f"Estimated r1: {r1_est}, Estimated r2: {r2_est}")

    # class KFConfig:
    #     def __init__(self, n_samples, t_step, f_s):
    #         self.N, self.T, self.Fs = n_samples, t_step, f_s
    #         self.q1 = np.array([1, 0, 0, 0])
    #         self.cov_w = np.eye(6) * 1e-4
    #         self.cov_lnk = np.eye(3) * 1e-2
    #         self.if_cheat_dw = self.if_cheat_w = self.if_cheat_a = False
    #         self.run_dynamic_update = self.run_measurement_update = True
    #     def __getitem__(self, key): return getattr(self, key)

    params.cov_w = np.eye(6) * 1e-2
    params.cov_lnk = 0.35**2 * np.eye(3) *10

    kf_input = {
        'gyr_1': gyr1, 'gyr_2': gyr2, 'acc_1': acc1, 'acc_2': acc2,
        'r1': r1_est, 'r2': r2_est
    }
    # only calculate performance after 30 seconds
    # N_converge = 3000

    q1_all, q2_all, _ = process_orientation_KF_Gframe(kf_input)

    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)
    angles_rad = qmt.eulerAngles(q_rel, axes='zyx')[:, 0]
    angle_deg = np.degrees(np.unwrap(angles_rad))

    # shift_converge, axis_converge, sign_converge, corr = find_best_shift_and_axis(angle_deg[N_converge], knee_gt[N_converge: ])
    shift, axis, sign, corr = find_best_shift_and_axis(angle_deg, knee_gt)

    est_final, gt_final = align_signals(angle_deg, knee_gt, shift, axis, sign)
    plt.plot(est_final, label='KF_Gframe with auto r')
    plt.plot(gt_final, label='Ground Truth')
    plt.legend()
    plt.title('Knee Angle Estimation using KF_Gframe with auto r')
    plt.xlabel('Samples')
    plt.ylabel('Knee Angle (degrees)')
    plt.savefig('plots/kf_gframe_auto_r_time_series.png', dpi=300)
    plt.close()
    errors_dict['kf_gframe_auto_r'] = np.abs(gt_final - est_final)
    print(f"KF_Gframe with auto r - RMSE: {np.sqrt(np.mean((gt_final - est_final)**2)):.2f} degrees")
    print(f"KG after convergence - RMSE: {np.sqrt(np.mean((gt_final[3000:] - est_final[3000:])**2)):.2f} degrees")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current Working Directory set to: {os.getcwd()}")
    parser = argparse.ArgumentParser(description='Run Knee Estimation')
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting')
    parser.add_argument('--method', type=str, default='all', choices=['vqf_olsson', 'heading_correction', 'opensense', 
                                 'vqf_olsson_heading_correction', 'kf_gframe', 'all'], help='Method to run')
    args = parser.parse_args()
    
    Path('plots').mkdir(exist_ok=True)

    data = prepare_data()
    errors_dict = {}

    jhat_femur = None
    jhat_tibia = None

    if args.method in ['kf_gframe', 'all']:
        run_kf_gframe_with_auto_r(data, args, errors_dict)
    
    if args.method in ['vqf_olsson', 'heading_correction', 'all']:
        jhat_femur, jhat_tibia = estimate_joint_axes(data)

    if args.method in ['vqf_olsson', 'all']:
        best_femur, best_tibia, est_final_o, gt_final_o = run_vqf_olsson(data, args, errors_dict, jhat_femur, jhat_tibia)
        jhat_femur = best_femur
        jhat_tibia = best_tibia

    if args.method in ['vqf_olsson_heading_correction', 'all']:
        run_vqf_olsson_heading_correction(data, args, errors_dict)


    if args.method in ['opensense', 'all']:
        run_opensense(data, args, errors_dict)


    if errors_dict:
        plot_time_series_error(errors_dict, no_plot=args.no_plot, num_entries=3)
        plot_error_comparison(errors_dict, no_plot=args.no_plot)

if __name__ == "__main__":
    main()
