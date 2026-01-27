import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signal # Required for Ctrl+C handling
import sys # Required for exit
import pandas as pd # Explicitly imported for type checking if needed

# --- Project Imports ---
from loaddata import data_load_process
from constants import FS, T
from calTools import alignDataE, quatmultiply

# --- Import Estimation Modules ---

from KF_Gframe import process_orientation_KF_Gframe as estimate_KF_base

# Other imports like orient_KF_IMUwBest, orient_KF_ive are available if needed.

# --- Import Visualization\Analysis Helpers ---
from visTools import (
    getRelHat, plotQEstReal, plot_orient, computeOrientMeanStd, 
    getDistDiff, getDFEulerDiff_ive,getDFEulerDiff, getDistDiff_ive
)

from calTools import lowpass_filter
# ==============================================================================
# I. SIGNAL HANDLER & SETUP
# ==============================================================================

# Global flag to signal termination
TERMINATE_FLAG = False
N_CONVERGE = 3000 # Samples to skip for post-convergence analysis
def signal_handler(sig, frame):
    """Handles Ctrl+C (SIGINT) signal to gracefully stop optimization loops."""
    global TERMINATE_FLAG
    print("\n--- TERMINATION SIGNAL RECEIVED (Ctrl+C) ---")
    print("Optimization will stop gracefully after the current iteration.")
    TERMINATE_FLAG = True
    # For hard exit, use: sys.exit(0)

# Register the signal handler when the script starts
signal.signal(signal.SIGINT, signal_handler)

# --- Helper functions for plotting and file management (Cleaned) ---

def prepare_output_folders(config):
    """Creates all necessary output directories (log, csv, fig) for a given file name."""
    log_folder = config['LOG_FOLDER']
    file_name = config['FILE_NAME_BASE']
    
    # Create file-specific paths
    PATH = os.path.join(log_folder, file_name)
    csv_path = os.path.join(PATH, "csv")
    fig_path = os.path.join(PATH, "fig")
    
    for folder in [log_folder, PATH, csv_path, fig_path]:
        os.makedirs(folder, exist_ok=True)
            
    return PATH + os.sep

def plot_angular_dist(angular_dist, path_to_save, if_save=False):
    """Plots angular distance and either shows or saves the figure."""
    plt.figure()
    plt.plot(angular_dist)
    plt.xlabel('Samples #')
    plt.ylabel('Angular distance (°)')
    plt.title('Angular Distance Error')
    if if_save is True:
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()

def trim_data_dict(data, N_samples_to_skip):
    """Trims time-series data fields in the dictionary for qREF, etc."""
    trimmed_data = {}
    for key, value in data.items():
        if key == "qREF":
                # Always trim qREF
                # print("Trimming qREF data..."
                # )
                # print(value.shape)
                # exit()
                trimmed_data[key] = value[N_samples_to_skip:]
        
        elif isinstance(value, np.ndarray) and value.ndim > 1 and value.shape[0] > N_samples_to_skip:
            trimmed_data[key] = value[N_samples_to_skip:]
        else:
            # Keep scalar, r1/r2, or already-trimmed 1D arrays
            trimmed_data[key] = value
    return trimmed_data

def trim_estimation_results(orientation_s1, orientation_s2, N_samples_to_skip):
    """Trims orientation estimation results."""
    # print(orientation_s1.shape)
    # exit()
    if orientation_s1.shape[0] > N_samples_to_skip:
        orientation_s1_trimmed = orientation_s1[N_samples_to_skip:]
    else:
        orientation_s1_trimmed = orientation_s1
        
    if orientation_s2.shape[0] > N_samples_to_skip:
        orientation_s2_trimmed = orientation_s2[N_samples_to_skip:]
    else:
        orientation_s2_trimmed = orientation_s2
        
    return orientation_s1_trimmed, orientation_s2_trimmed

import seaborn as sns


def analyze_and_log(data, N, orientation_s1, orientation_s2, r1_opt, r2_opt, 
                    filename, method, PATH, save_csv=False, suffix=""):
    """
    Performs full analysis (for Raw results). Calculates Full-curve metrics 
    and Post-Convergence metrics (trimming internally from N_CONVERGE).
    """
    full_name = filename + suffix
    
    # 1. Relative orientation calculation
    rel_hat = getRelHat(orientation_s1, orientation_s2)
    
    # 2. Distance and Difference calculations (Full Curve)
    df_euler_diff = getDFEulerDiff_ive(data, rel_hat, PATH + "csv/", full_name, save_csv)
    angular_dist = getDistDiff_ive(data, N, rel_hat, PATH, full_name, save_csv)
    
    # -----------------------------------------------------------
    # Slice Data for Post-Convergence Analysis (Starts at N_CONVERGE)
    # -----------------------------------------------------------
    
    N_start_analysis = min(N_CONVERGE, N - 1)
    
    df_euler_diff_post = df_euler_diff.iloc[N_start_analysis:]
    angular_dist_post = angular_dist[N_start_analysis:]
    
    # -----------------------------------------------------------
    # 3. Plots (Using Full Data)
    # -----------------------------------------------------------
    plotQEstReal(rel_hat, data["qREF"])
    plot_angular_dist(angular_dist, PATH + "fig/" + full_name + "_angdist.png", if_save=save_csv)
    plot_orient(orientation_s1, orientation_s2, data["qGS1_ref"], data["qGS2_ref"], PATH + "fig/" + full_name + "_qEst")

    # -----------------------------------------------------------
    # 4. Metrics (Full Curve)
    # -----------------------------------------------------------
    mean_dist_full = angular_dist.mean()
    std_dist_full = angular_dist.std()
    mae_axis_full = df_euler_diff.abs().mean() * (180.0 / np.pi)
    rmse_axis_full = np.sqrt((df_euler_diff**2).mean()) * (180.0 / np.pi)
    
    # -----------------------------------------------------------
    # 5. Metrics (Post-Convergence)
    # -----------------------------------------------------------
    
    mean_dist_post = angular_dist_post.mean()
    std_dist_post = angular_dist_post.std()
    mae_axis_post = df_euler_diff_post.abs().mean() * (180.0 / np.pi)
    rmse_axis_post = np.sqrt((df_euler_diff_post**2).mean()) * (180.0 / np.pi)
    
    # -----------------------------------------------------------
    # 6. Console Output
    # -----------------------------------------------------------

    print(f"\n--- {method}{suffix} RESULTS (N={N}) ---")
    print(f"Optimized r1, r2: {r1_opt.T.flatten()} | {r2_opt.T.flatten()}")
    
    print("\n[FULL CURVE Metrics]")
    print(f"Mean Angular Dist: {mean_dist_full:.4f}°, Std: {std_dist_full:.4f}°")
    
    print(f"\n[POST-CONVERGENCE Metrics (Samples {N_start_analysis} to {N-1})]")
    print(f"Mean Angular Dist: {mean_dist_post:.4f}°, Std: {std_dist_post:.4f}°")
    print(f"Post-Conv RMSE (deg): R:{rmse_axis_post.iloc[0]:.4f} | P:{rmse_axis_post.iloc[1]:.4f} | Y:{rmse_axis_post.iloc[2]:.4f}")
    
    computeOrientMeanStd(r1_opt, r2_opt, df_euler_diff, angular_dist, full_name)
    
    # -----------------------------------------------------------
    # 7. Save Final Trajectory NPZ 
    # -----------------------------------------------------------
    np.savez_compressed(
        os.path.join(PATH, f"traj_{method}{suffix}.npz"),
        orientation_s1=orientation_s1,
        orientation_s2=orientation_s2,
        r1_opt=r1_opt,
        r2_opt=r2_opt,
        angular_dist_full=angular_dist,
        angular_dist_post=angular_dist_post
    )

    # -----------------------------------------------------------
    # 8. Return dictionary for the summary table (Post-Convergence)
    # -----------------------------------------------------------
    return {
        'mean_full': mean_dist_full, 
        'std_full': std_dist_full,
        'mae_full': mae_axis_full,
        'rmse_full': rmse_axis_full,
        'mean_post': mean_dist_post,
        'std_post': std_dist_post,
        'mae_post': mae_axis_post,
        'rmse_post': rmse_axis_post,
    }

def run_analysis_suite(data, N, orientation_s1, orientation_s2, rel_hat_input, r1_opt, r2_opt, 
                       filename, method, PATH, save_csv=False, suffix=""):
    """
    Performs full analysis suite using the provided relative quaternion (rel_hat_input).
    
    *** ASSUMPTION: The input data here is ALREADY TRIMMED, so analysis starts at index 0. ***
    """
    full_name = filename + suffix
    
    # 1. Relative orientation is passed as rel_hat_input
    rel_hat = rel_hat_input
    N_samples = rel_hat_input.shape[0]
    # 2. Distance and Difference calculations (Full Curve = Post-Conv Curve)
    # df_euler_diff = getDFEulerDiff_ive(data, rel_hat, PATH + "csv/", full_name, save_csv)
    # angular_dist = getDistDiff_ive(data, N_samples, rel_hat, PATH, full_name, save_csv)
    
    df_euler_diff = getDFEulerDiff(data, rel_hat, PATH + "csv/", full_name, save_csv)
    angular_dist = getDistDiff(data, N_samples, rel_hat, PATH, full_name, save_csv)

    # -----------------------------------------------------------
    # Since input is trimmed, N_start_analysis = 0
    # -----------------------------------------------------------
    N_start_analysis = 0
    df_euler_diff_post = df_euler_diff.iloc[N_start_analysis:]
    angular_dist_post = angular_dist[N_start_analysis:]
    
    # -----------------------------------------------------------
    # 3. Plots (Using Input Data)
    # -----------------------------------------------------------
    print("Generating Plots...")
    print("rel_hat shape:", rel_hat.shape)

    plotQEstReal(rel_hat, data["qREF"])
    print("Plotted QEst vs Real."
          )
    print("Angular dist shape:", angular_dist.shape)
    plot_angular_dist(angular_dist, PATH + "fig/" + full_name + "_angdist.png", if_save=save_csv)
    # Note: plot_orient is typically used for full trajectory, but we call it here for consistency
    plot_orient(orientation_s1, orientation_s2, data["qGS1_ref"], data["qGS2_ref"], PATH + "fig/" + full_name + "_qEst") 

    # -----------------------------------------------------------
    # 4. Metrics (Full Curve / Post-Convergence)
    # -----------------------------------------------------------
    
    # Metrics are identical since analysis starts at index 0
    mean_dist_full = angular_dist.mean()
    std_dist_full = angular_dist.std()
    mae_axis_full = df_euler_diff.abs().mean() * (180.0 / np.pi)
    rmse_axis_full = np.sqrt((df_euler_diff**2).mean()) * (180.0 / np.pi)
    
    mean_dist_post = angular_dist_post.mean()
    std_dist_post = angular_dist_post.std()
    mae_axis_post = df_euler_diff_post.abs().mean() * (180.0 / np.pi)
    rmse_axis_post = np.sqrt((df_euler_diff_post**2).mean()) * (180.0 / np.pi)

    # -----------------------------------------------------------
    # 6. Console Output
    # -----------------------------------------------------------

    print(f"\n--- {method}{suffix} RESULTS (N={N} trimmed samples) ---")
    print(f"Optimized r1, r2: {r1_opt.T.flatten()} | {r2_opt.T.flatten()}")
    
    print("\n[TRIMMED Curve Metrics (Equivalent to Post-Convergence)]")
    print(f"Mean Angular Dist: {mean_dist_full:.4f}°, Std: {std_dist_full:.4f}°")
    
    print(f"RMSE (deg): R:{rmse_axis_post.iloc[0]:.4f} | P:{rmse_axis_post.iloc[1]:.4f} | Y:{rmse_axis_post.iloc[2]:.4f}")
    
    computeOrientMeanStd(r1_opt, r2_opt, df_euler_diff, angular_dist, full_name)

    # -----------------------------------------------------------
    # 8. Return dictionary for the summary table (Post-Convergence)
    # -----------------------------------------------------------
    return {
        'mean_full': mean_dist_full, # Use 'full' for the trimmed metric
        'std_full': std_dist_full,
        'mae_full': mae_axis_full,
        'rmse_full': rmse_axis_full,
        'mean_post': mean_dist_post,
        'std_post': std_dist_post,
        'mae_post': mae_axis_post,
        'rmse_post': rmse_axis_post,
    }

def analyze_estimation_method_align(data, N, orientation_s1_est, orientation_s2_est, r1_opt, r2_opt, 
                              filename, method, PATH, save_csv=False, correct_misalignment=False, misaligned_quat=None):
    """
    Wrapper that performs a single relative alignment and runs analysis for raw and aligned results.
    
    *** NOTE: This function now expects all inputs to be the TRIMMED set (after sample N_CONVERGE). ***
    """
    print(f"\n==================== Running ALIGNED ANALYSIS for {method} (Trimmed Input) ====================")
    
    # 1. Calculate RAW Relative Orientation (The quaternion to be aligned)
    # Note: This is calculated from the TRIMMED inputs
    rel_hat_raw = getRelHat(orientation_s1_est, orientation_s2_est)
    
    # --- 2. MISALIGNMENT CORRECTION (Aligning Relative Hat to Reference qREF) ---
    print(f"Aligning RAW Relative trajectory to Reference (qREF)...")
    
    # Align the RAW RELATIVE HAT against the trimmed ground truth relative quaternion (data["qREF"])
    _, rel_hat_aligned, qMS_est, qVI_est = alignDataE(rel_hat_raw, data["qREF"])

    if correct_misalignment and (misaligned_quat is not None):
        print("Applying additional misalignment correction...")
        q_mis = misaligned_quat / np.linalg.norm(misaligned_quat)
        # Apply this misalignment to the already aligned relative hat
        rel_hat_aligned = np.array([quatmultiply(q_mis, q) for q in rel_hat_aligned])
    
    # --- 3. ALIGNED ANALYSIS ---
    print(f"\n--- Running ALIGNED Analysis for {method} ---")
    print(f"Relative Misalignment (qMS/qVI): {qMS_est.flatten()} | {qVI_est.flatten()}")
    
    # run_analysis_suite now runs the calculation on the trimmed data starting from index 0
    result_aligned = run_analysis_suite(data, N, orientation_s1_est, orientation_s2_est, rel_hat_aligned, r1_opt, r2_opt, 
                                        filename, method, PATH, save_csv, "_aligned")
    
    return result_aligned # Returns dictionary with stats
    # return result_aligned, orientation_s1_est, orientation_s2_est # Returns dictionary with stats


# ==============================================================================
# III. CONFIGURATION & MAIN EXECUTION
# ==============================================================================


# --- Configuration Dictionary ---
CONFIG = {
    # Data Loading Parameters
    "TRUNCATE": 60000,
    "DATA_PATH_NPZ": r"C:\Users\ruiyuanli\OneDrive - Delft University of Technology\work\project\second_paper\python_driftfree_imuPosition\data_ive\data1208\Vicon\S4\S4.mat",
    "SAVE_CSV": False,
    "FORCE_REPROCESS": False,
    # Filtering/Optimization Parameters
    "FS": 100,
    "ITERATION": 5,
    "CW_LNK": (0.35*0.35)*10,
    "CW_W": 1e-2 ,

    # handle the misalignment
    "CORRECT_MISALIGNMENT": False,
    "MISALIGNED_QUAT": np.array([-0.438, 0.539, -0.715, -0.083]),
    
    # Output File Naming
    "EXPERIMENT_NAME": "CLEAN_RUN_S13_W1",
    "FILE_NAME_BASE": "S13_W1_ANALYSIS", 
    "LOG_FOLDER": f"log/CLEAN_RUN_cwLnk{1e-1 * 1:.2f}_cwW{1e-1 * 0.1:.2f}/" ,

    # if r -
    "IF_MINUS_R": False,
    # "IF_IMU_LOWPASS": True,
    "IF_IMU_LOWPASS": False,

    "LOWPASSCUTOFF": 6.0,  # Hz
}

def setup_global_params(config):
    """Prints config params (legacy function, no longer sets global state)."""
    print(f"Config: Fs={config['FS']}, T={1/config['FS']}, Cov_lnk={config['CW_LNK']}")




def main(config):
    global TERMINATE_FLAG, N_CONVERGE
    
    # 1. Setup Environment and Load Data
    setup_global_params(config)
    PATH = prepare_output_folders(config)
    
    print(f"\nAttempting to load data from: {config['DATA_PATH_NPZ']}")
    # Load and process FULL data
    data_full = data_load_process(config['TRUNCATE'], config['DATA_PATH_NPZ'], config['FORCE_REPROCESS'])

    
    if data_full is None:
        print("Data loading failed. Exiting.")
        return

    N_full = data_full['gyr_1'].shape[1]

    r1_opt_true = data_full["r1"]
    r2_opt_true = data_full["r2"]
    
    print(f"Data successfully loaded. N_full={N_full} timesteps.")
    
    # --- Pre-Trim Data for Aligned Analysis Inputs ---
    print(f"\nPreparing trimmed data set for aligned analysis (skipping first {N_CONVERGE} samples)...")
    data_trimmed = trim_data_dict(data_full, N_CONVERGE)
    N_trimmed = data_trimmed['gyr_1'].shape[0] # New length
    
    # --- Results Dictionary to store performance metrics ---
    results_summary = {}



    # 3. Kalman Filter (KF) Estimation (Run on FULL data)
    if not TERMINATE_FLAG:
        print("\n==================== Running KF Base Estimation (Full Data) ====================")

        orientation_s1_base_kf, orientation_s2_base_kf, P_list = estimate_KF_base(data_full)
        
        # Capture metrics (Raw, uses internal trimming logic in analyze_and_log)
        res_kf_base = analyze_and_log(
            data_full, N_full, orientation_s1_base_kf, orientation_s2_base_kf, 
            r1_opt_true, r2_opt_true, 
            config['FILE_NAME_BASE'], "KF_Base", PATH, config['SAVE_CSV']
        )
        results_summary["KF Base (Raw)"] = res_kf_base
        
        # *** Aligned Analysis: Trimming input data before alignment/analysis ***
        print("\n==================== Running KF Base Aligned (Trimmed Data) ====================")
        # 1. Trim KF outputs
        orientation_s1_base_kf_trimmed, orientation_s2_base_kf_trimmed = trim_estimation_results(orientation_s1_base_kf, orientation_s2_base_kf, N_CONVERGE)
        
        # 2. Run Aligned Analysis on the trimmed data set
        res_kf_base_aligned = analyze_estimation_method_align(
            data_trimmed, N_trimmed, # Use TRIMMED data (qREF is trimmed) and N_trimmed
            orientation_s1_base_kf_trimmed, orientation_s2_base_kf_trimmed, # Use TRIMMED orientations
            r1_opt_true, r2_opt_true,
            config['FILE_NAME_BASE'], "KF_Base", PATH, config['SAVE_CSV'],
            config['CORRECT_MISALIGNMENT'], config['MISALIGNED_QUAT']
        )
        results_summary["KF Base (Aligned)"] = res_kf_base_aligned
    

        
    if TERMINATE_FLAG:
        print("\nExecution terminated by user (Ctrl+C). Skipping remaining tasks.")

    # --- PRINT FINAL SUMMARY TABLES ---
    
    # 1. Overall Angular Error Table
    print("\n" + "="*80)
    print(f"{'OVERALL ANGULAR ERROR SUMMARY':^80}")
    print("="*80)
    # Note: For Raw, 'Full' means the full curve. For Aligned, 'Full' means the trimmed curve (N_CONVERGE onwards)
    print(f"{'METHOD':<25} | {'MEAN (Full/Trim, °)':<15} | {'STD (Full/Trim, °)':<15} | {'MEAN (Post-Conv, °)':<19}")
    print("-" * 80)
    
    expected_methods = [
        "Dead Reckoning", "Dead Reckoning (Aligned)",
        "KF Base (Raw)", "KF Base (Aligned)",
        "OPT Base (Raw)", "OPT Base (Aligned)",
        "KF IMU (Raw)", "KF IMU (Aligned)", 
        "OPT IMU (Raw)", "OPT IMU (Aligned)",
        "KF IMU_wb (Raw)", "KF IMU_wb (Aligned)",
        "KF Michael (raw)", "KF Michael (Aligned)"
    ]
    
    for method in expected_methods:
        if method in results_summary:
            res = results_summary[method]
            print(f"{method:<25} | {res['mean_full']:<15.4f} | {res['std_full']:<15.4f} | {res['mean_post']:<19.4f}")
        else:
            print(f"{method:<25} | {'N/A':<15} | {'N/A':<15} | {'N/A':<19}")
            
    print("="*80 + "\n")

    # 2. Per-Axis Error Table (Post-Convergence)
    print("\n" + "="*110)
    print(f"{'PER-AXIS ERROR SUMMARY (Post-Convergence, deg)':^110}")
    print("="*110)
    # Header row
    print(f"{'METHOD':<25} | {'ROLL MAE':<10} | {'PITCH MAE':<10} | {'YAW MAE':<10} | {'ROLL RMSE':<10} | {'PITCH RMSE':<10} | {'YAW RMSE':<10}")
    print("-" * 110)

    for method in expected_methods:
        if method in results_summary:
            res = results_summary[method]
            mae = res['mae_post']   # Series: [roll_diff, pitch_diff, yaw_diff]
            rmse = res['rmse_post'] # Series: [roll_diff, pitch_diff, yaw_diff]
            
            # Use .iloc[0], .iloc[1], .iloc[2]
            print(f"{method:<25} | {mae.iloc[0]:<10.4f} | {mae.iloc[1]:<10.4f} | {mae.iloc[2]:<10.4f} | {rmse.iloc[0]:<10.4f} | {rmse.iloc[1]:<10.4f} | {rmse.iloc[2]:<10.4f}")
        else:
            print(f"{method:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    print("="*110 + "\n")



if __name__ == "__main__":
    # The absolute path below should be updated by the user for local execution.

    os.path.dirname(os.path.abspath(__file__))
    print("Current working directory:", os.getcwd())
    # make the working path where the script is
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    CONFIG["DATA_PATH_NPZ"] = r".\Subject08\walking\WITH_R1R2_rawIMU_rawMocap_noLPF\synced_2_femur_r_imu_tibia_r_imu.npz"
    CONFIG["DATA_PATH_NPZ"] = r".\Subject08\walking\WITH_R1R2_rawIMU_rawMocap_noLPF\synced_4_tibia_r_imu_calcn_r_imu.npz"


    CONFIG["FORCE_REPROCESS"] = False
    
    CONFIG["CORRECT_MISALIGNMENT"] = False
    main(CONFIG)
