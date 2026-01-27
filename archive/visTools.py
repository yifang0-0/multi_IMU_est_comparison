from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calTools import quaternion_to_euler, angular_distance, quatmultiply, quatconj, quatinv, quatnormalize
from scipy.spatial.transform import Rotation as R
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_dual_window_kf_direct(r1, r2, sub_id, walk_iter, joint_name_str, fig_path, start_idx=0, end_idx=None):
    """
    Plots r1 and r2 estimates directly from arrays.
    
    Parameters:
    r1, r2 (np.array): Estimation arrays of shape (N, 3)
    motion_type (str): Name of the motion (e.g., 'walking')
    sub_id (int/str): Subject identifier
    walk_iter (int): Iteration number
    joint_name_str (str): Anatomical joint name (e.g., 'right hip')
    base_   (str): Directory where the figure will be saved
    """
    # Calculate magnitudes
    mag1 = np.linalg.norm(r1, axis=1)
    mag2 = np.linalg.norm(r2, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    def plot_helper(ax, signal, magnitude, start, end, title):
        # Slice data
        s_slice = signal[start:end]
        m_slice = magnitude[start:end]
        t = np.arange(start, start + len(s_slice))
        
        ax.plot(t, s_slice, alpha=0.7)
        ax.plot(t, m_slice, 'k--', linewidth=1.5, label="Magnitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if start == start_idx: 
            ax.set_ylabel("Component Value")
        ax.legend(["X", "Y", "Z", "Mag"], fontsize='small', loc='upper right')

    # --- LEFT COLUMN: USER DEFINED WINDOW ---
    plot_helper(axes[0, 0], r1, mag1, start_idx, end_idx, f"r1: Window [{start_idx}:{end_idx if end_idx else 'End'}]")
    plot_helper(axes[1, 0], r2, mag2, start_idx, end_idx, f"r2: Window [{start_idx}:{end_idx if end_idx else 'End'}]")

    # --- RIGHT COLUMN: LAST 500 SAMPLES ---
    last_start = max(0, len(r1) - 500)
    plot_helper(axes[0, 1], r1, mag1, last_start, None, "r1: Last 500 Samples (Stability)")
    plot_helper(axes[1, 1], r2, mag2, last_start, None, "r2: Last 500 Samples (Stability)")

    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 1].set_xlabel("Sample Index")
    
    plt.suptitle(f"Subject {sub_id} | Joint {walk_iter} ({joint_name_str}) - Rotation Stability", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Define save path based on metadata
    # path_fig = os.path.join(base_path, motion_type, f"Subject{sub_id}", f"walk_{walk_iter}")
    # if not os.path.exists(path_fig):
        # os.makedirs(path_fig)
        
    save_name = f"KF_r1r2_windowed_sub{sub_id}_walk{walk_iter}.png"
    plt.savefig(os.path.join(fig_path, save_name))
    plt.show()


# %%

def plot_r_analysis_comprehensive(r1_list, r2_list, r1_true, r2_true, fig_path, sub_id, joint_iter, n_converge=None):
    """
    Comprehensive visualization including:
    1. 3D Trajectory Path (Full)
    2. Vector Norms (Full Length)
    3. Split View (First/Last 500) of Euclidean Error Norms
    """
    r1_arr = np.array(r1_list)
    r2_arr = np.array(r2_list)
    n_total = len(r1_arr)
    
    # --- 1. Data Processing ---
    # Euclidean Norms (Lengths)
    est_r1_norm = np.linalg.norm(r1_arr, axis=1)
    est_r2_norm = np.linalg.norm(r2_arr, axis=1)
    true_r1_norm = np.linalg.norm(r1_true)
    true_r2_norm = np.linalg.norm(r2_true)
    
    # Euclidean Error (Distance to target)
    err1_norm = np.linalg.norm(r1_arr - r1_true.flatten(), axis=1)
    err2_norm = np.linalg.norm(r2_arr - r2_true.flatten(), axis=1)

    # --- 2. Setup Figure ---
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # Subplot 1: 3D Trajectory (Stays Full Length)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    def add_path(ax, points, cmap_name):
        segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
        norm = plt.Normalize(0, len(segments))
        lc = Line3DCollection(segments, cmap=plt.get_cmap(cmap_name), norm=norm)
        lc.set_array(np.arange(len(segments)))
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
    
    add_path(ax1, r1_arr, "viridis")
    add_path(ax1, r2_arr, "plasma")
    ax1.scatter(*r1_true.flatten(), color='black', marker='o', s=100, label='r1 True')
    ax1.scatter(*r2_true.flatten(), color='red', marker='^', s=100, label='r2 True')
    ax1.set_title("3D Convergence Path")
    ax1.legend()

    # Subplot 2: Vector Norm Comparison (Full Length)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(est_r1_norm, label="Est |r1|", color='tab:blue')
    ax2.axhline(y=true_r1_norm, color='navy', linestyle='--', label="True |r1|")
    ax2.plot(est_r2_norm, label="Est |r2|", color='tab:orange')
    ax2.axhline(y=true_r2_norm, color='darkorange', linestyle='--', label="True |r2|")
    ax2.set_title("Full Signal: Vector Norms (m)")
    ax2.set_xlabel("Sample Index")
    ax2.grid(True, linestyle=':')
    ax2.legend(loc='upper right', ncol=2)

    # Subplot 3 & 4: Error Norm Split View (First/Last 500)
    ax_start = fig.add_subplot(gs[1, 0:2]) # Bottom left/middle
    ax_end = fig.add_subplot(gs[1, 2], sharey=ax_start) # Bottom right
    
    # Plot Start (First 500)
    idx_s = np.arange(0, min(500, n_total))
    ax_start.plot(idx_s, err1_norm[idx_s], color='tab:blue', label="|r1_err|")
    ax_start.plot(idx_s, err2_norm[idx_s], color='tab:orange', label="|r2_err|")
    if n_converge:
        
        ax_start.axvline(x=min(n_converge, min(500, n_total)), color='red', linestyle='--', label='Convergence Pt')

    ax_start.set_title(f"Error Norm: First 500 Samples sub:{sub_id}, joint pair:{joint_iter}")
    ax_start.set_xlabel("Sample Index")
    ax_start.set_ylabel("Error (m)")
    ax_start.grid(True, linestyle=':')
    ax_start.legend()

    # Plot End (Last 500)
    idx_e = np.arange(max(0, n_total - 500), n_total)
    ax_end.plot(idx_e, err1_norm[idx_e], color='tab:blue')
    ax_end.plot(idx_e, err2_norm[idx_e], color='tab:orange')
    ax_end.set_title("Error Norm: Last 500 Samples")
    ax_end.set_xlabel("Sample Index")
    ax_end.grid(True, linestyle=':')
    plt.setp(ax_end.get_yticklabels(), visible=False)

    if "walking" in fig_path.lower():
        fig.suptitle(f"Subject {sub_id} | Joint {joint_iter} - Method: (Walking)", fontsize=16)
    else:
        fig.suptitle(f"Subject {sub_id} | Joint {joint_iter} - Method: complexTasks ", fontsize=16) 
    # --- 3. Save and Close ---
    # fig.suptitle(f"Analysis for Me", fontsize=16, fontweight='bold')
    save_file = os.path.join(fig_path,  f"r_comprehensive.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()



def plot_angular_dist(angular_dist, PATH, if_save = False):
    # Compute angular distance
    plt.figure()
    plt.plot(angular_dist)
    plt.xlabel('Samples # (Fs:50Hz)')
    plt.ylabel('Angular distance (°)')
    if if_save is True:
        plt.savefig(PATH)
        plt.close()
    else:
        plt.show()

def computeOrientMeanStd(r1, r2, df, angular_dist, filename):
    """
    Computes mean and standard deviation for roll, pitch, and yaw differences.

    Args:
        df (pd.DataFrame): Input DataFrame with columns ["roll_diff", "pitch_diff", "yaw_diff"]
        filename (str): Name to be used as the row index.

    Returns:
        pd.DataFrame: A single-row DataFrame with 6 columns for mean and std values, indexed by filename.
    """
    mean_values = df.mean().rename(lambda x: x + "_mean")  # Compute mean and rename
    std_values = df.std().rename(lambda x: x + "_std")  # Compute std and rename
    
    angular_dist_mean = np.mean(angular_dist)
    angular_dist_std = np.std(angular_dist)
    
    # Convert r1 and r2 (which are lists) into a DataFrame-compatible format
    r1_dict = {f"r1_{ chr(ord('x') + i)}": r1.reshape(-1)[i] for i in range(len(r1.reshape(-1)))}
    r2_dict = {f"r2_{ chr(ord('x') + i)}": r2.reshape(-1)[i] for i in range(len(r2.reshape(-1)))}
    
    # Combine all values into a single DataFrame row
    result_df = pd.DataFrame(
        [mean_values.tolist() + std_values.tolist() + list(r1_dict.values()) + list(r2_dict.values())+[angular_dist_mean, angular_dist_std]],  
        columns=mean_values.index.tolist() + std_values.index.tolist() + list(r1_dict.keys()) + list(r2_dict.keys())+["angular_dist_mean", "angular_dist_std"],
        index=[filename]  # Set row index to filename
    )
    result_df.to_csv(filename+"_result_df.csv")
    return result_df

def plotQEstReal(est, real, filename = None):
    # Compute angular distance
    real_df = getDataEulerDgr(real)
    roll_true  = real_df["roll"]
    pitch_true = real_df["pitch"]
    yaw_true   = real_df["yaw"]
    
    est_df = getDataEulerDgr(est)
    
    roll_est  = est_df["roll"]
    pitch_est = est_df["pitch"]
    yaw_est   = est_df["yaw"]
        
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Roll
    axs[0].plot(roll_true, label='roll_real', color='blue')
    axs[0].plot(roll_est, label='roll_est', color='blue', linestyle='--')
    axs[0].set_ylabel('Roll (°)')
    axs[0].legend()
    axs[0].grid(True)

    # Pitch
    axs[1].plot(pitch_true, label='pitch_real', color='green')
    axs[1].plot(pitch_est, label='pitch_est', color='green', linestyle='--')
    axs[1].set_ylabel('Pitch (°)')
    axs[1].legend()
    axs[1].grid(True)

    # Yaw
    axs[2].plot(yaw_true, label='yaw_real', color='red')
    axs[2].plot(yaw_est, label='yaw_est', color='red', linestyle='--')
    axs[2].set_ylabel('Yaw (°)')
    axs[2].set_xlabel('Samples # (Fs: 50Hz)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename+"_agdis.png")
        plt.close()
    # plt.savefig(filename, dpi=300)
    plt.close()

def getDataEulerRad(data):
    EulerRad = quaternion_to_euler(data)
    df = pd.DataFrame(EulerRad.T, columns=["roll", "pitch", "yaw"])
    return df

def getDataEulerDgr(data):
    EulerRad = quaternion_to_euler(data)
    df = pd.DataFrame((EulerRad * 180 / np.pi).T, columns=["roll", "pitch", "yaw"])
    return df

def getDFEulerDiff(data, rel_hat, PATH, filename, if_save=False):
    dim = data['qREF'].shape[0]
    qRef = data["qREF"]

    quat_diff = np.zeros((dim,4))
    for t in range(dim):
        quat_diff[t,:] = quatmultiply(quatinv(qRef[t,:]), rel_hat[t,:])

        if quat_diff[t,0]<0:
            quat_diff[t] = -quat_diff[t]

        # Get the angle from the quaternion difference
        if quat_diff[t,0]**2>1:
                # print("Quaternion norm is greater than 1, normalizing.")
                quat_diff[t] = quatnormalize(quat_diff[t])

    euler_diff_rad = R.from_quat(quat_diff[:, [1, 2, 3, 0]]).as_euler('xyz')
    euler_diff_rad = (euler_diff_rad + np.pi) % (2 * np.pi) - np.pi
    # euler_diff = rel_eu.T - qRef_eu.T[:N,:]
   

    # Save the result to CSV
    df = pd.DataFrame(euler_diff_rad, columns=["roll_diff", "pitch_diff", "yaw_diff"])
    if if_save==True: 
        csv_path = os.path.join(PATH, filename+"_eulerdiff.csv")
        if os.path.exists(PATH)==False:
            os.makedirs(PATH)
        df.to_csv(csv_path, index=False)
    # print("Mean absolute error (deg):")
    # print(df.abs().mean()*180/np.pi)

    df_deg = df.copy()
    df_deg[["roll_diff", "pitch_diff", "yaw_diff"]] = np.degrees(
        df[["roll_diff", "pitch_diff", "yaw_diff"]]
    )
    rmse = np.sqrt((df_deg**2).mean())
    # print("\nRMSE (deg) per axis:")
    # print(rmse)
    # Plot in degrees
    plt.figure(figsize=(10, 6))

    plt.plot(df_deg.index, df_deg["roll_diff"], label="Roll (°)", linewidth=1.5)
    plt.plot(df_deg.index, df_deg["pitch_diff"], label="Pitch (°)", linewidth=1.5)
    plt.plot(df_deg.index, df_deg["yaw_diff"], label="Yaw (°)", linewidth=1.5)

    plt.xlabel("Sample Index")
    plt.ylabel("Angle Difference (degrees)")
    plt.title("Roll, Pitch, Yaw Differences (in Degrees)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if if_save==True:
        plot_angular_dist_path = os.path.join(PATH, filename+"_eulerdiff.png")
        if os.path.exists(PATH)==False:
            os.makedirs(PATH)
        plt.savefig(plot_angular_dist_path)
    plt.show()
    return df

def getDFEulerDiff_ive(data, rel_hat, PATH, filename, if_save=False):
    qRef = data["qREF"]
    N = qRef.shape[0]
    rel_eu = quaternion_to_euler(rel_hat)
    qRef_eu = quaternion_to_euler(qRef)

    print("shape rel_eu:", rel_eu.shape)
    print("shape qRef_eu:", qRef_eu.shape)
        #plot the row [itch yaw]
        # Convert radians to degrees
    rel_eu_deg = np.degrees(rel_eu).T
    qRef_eu_deg = np.degrees(qRef_eu).T

    # Sample indices for x-axis
    samples = np.arange(rel_eu_deg.shape[0])

    # Plot Roll, Pitch, Yaw separately
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    angles = ["Roll", "Pitch", "Yaw"]

    # a=qRef_eu_deg[:,2]
    # qRef_eu_deg[:,2]=qRef_eu_deg[:,1]
    # qRef_eu_deg[:,1]=a


    for i, ax in enumerate(axes):
        ax.plot(samples, qRef_eu_deg[:, i], label=f"{angles[i]} Reference", linewidth=1.5)
        ax.plot(samples, rel_eu_deg[:, i], label=f"{angles[i]} Estimated", linewidth=1.5, linestyle="--")
        ax.set_ylabel("Angle (°)")
        ax.set_title(f"{angles[i]} Comparison")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.show()
    # end
    # print(rel_eu_deg.mean())

    euler_diff = rel_eu.T[1:, :] - qRef_eu.T[:-1, :]
    euler_diff = np.vstack([np.zeros((1, euler_diff.shape[1])), euler_diff])
    

    # Save the result to CSV
    df = pd.DataFrame(euler_diff, columns=["roll_diff", "pitch_diff", "yaw_diff"])
    if if_save==True: 
        if os.path.exists(PATH)==False:
            os.makedirs(PATH)
        df.to_csv(PATH+filename+"eulerdiff.csv", index=False)
    print("Mean absolute error (deg):")
    print(df.abs().mean()*180/np.pi)

    df_deg = df.copy()
    df_deg[["roll_diff", "pitch_diff", "yaw_diff"]] = np.degrees(
        df[["roll_diff", "pitch_diff", "yaw_diff"]]
    )
    rmse = np.sqrt((df_deg**2).mean())
    print("\nRMSE (deg) per axis:")
    print(rmse)
    # Plot in degrees
    plt.figure(figsize=(10, 6))

    plt.plot(df_deg.index, df_deg["roll_diff"], label="Roll (°)", linewidth=1.5)
    plt.plot(df_deg.index, df_deg["pitch_diff"], label="Pitch (°)", linewidth=1.5)
    plt.plot(df_deg.index, df_deg["yaw_diff"], label="Yaw (°)", linewidth=1.5)

    plt.xlabel("Sample Index")
    plt.ylabel("Angle Difference (degrees)")
    plt.title("Roll, Pitch, Yaw Differences (in Degrees)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    return df

def plot_orient(orientation_s1,orientation_s2,qGS1_ref,qGS2_ref, filename = None):
    qGS1_ref = qGS1_ref[:params.N,:]
    qGS2_ref = qGS1_ref[:params.N,:]
    
    euler_s1 = quaternion_to_euler(orientation_s1)*180/np.pi
    euler_s2 = quaternion_to_euler(orientation_s2)*180/np.pi
    euler_ref1 = quaternion_to_euler(qGS1_ref)*180/np.pi
    euler_ref2 = quaternion_to_euler(qGS2_ref)*180/np.pi
    # euler_s1 = euler_s1.T  # Now shape is (1000, 3)
    # euler_s2 = euler_s2.T
    # euler_ref1 = euler_ref1.T
    # euler_ref2 = euler_ref2.T

    t = np.arange(orientation_s1.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    titles = ['Sensor 1: Quaternion', 'Sensor 2: Quaternion', 
              'Sensor 1: Euler angles', 'Sensor 2: Euler angles']
    labels_q = ['w', 'x', 'y', 'z']
    labels_e = ['Roll', 'Pitch', 'Yaw']
    colors = ['r', 'g', 'b', 'm']

    # Plot quaternion components
    for i in range(4):  # w, x, y, z
        axs[0, 0].plot(t, orientation_s1[:, i], linestyle='--', color=colors[i], label=f'{labels_q[i]} est', alpha=0.5, linewidth=1)
        axs[0, 0].plot(t, qGS1_ref[:, i], linestyle='-', color=colors[i], label=f'{labels_q[i]} ref', linewidth=2)

        axs[0, 1].plot(t, orientation_s2[:, i], linestyle='--', color=colors[i], label=f'{labels_q[i]} est',alpha=0.5, linewidth=1)
        axs[0, 1].plot(t, qGS2_ref[:, i], linestyle='-', color=colors[i], label=f'{labels_q[i]} ref',  linewidth=2)

    # Plot Euler angles
    for i in range(3):
        axs[1, 0].plot(t, euler_s1[i], linestyle='--', color=colors[i], label=f'{labels_e[i]} est',alpha=0.5, linewidth=1)
        axs[1, 0].plot(t, euler_ref1[i], linestyle='-', color=colors[i], label=f'{labels_e[i]} ref', linewidth=2)

        axs[1, 1].plot(t, euler_s2[i], linestyle='--', color=colors[i], label=f'{labels_e[i]} est',alpha=0.5, linewidth=1)
        axs[1, 1].plot(t, euler_ref2[i], linestyle='-', color=colors[i], label=f'{labels_e[i]} ref', linewidth=2)

    for ax, title in zip(axs.flat, titles):
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if filename is not None:
        full_path = os.path.abspath(filename+"_orient.png")
        print("Path length:", len(full_path))
        os.makedirs(os.path.dirname(filename+"_orient.png"), exist_ok=True)
        try:
            plt.savefig(filename+"_orient.png")
        except:
            print("Error saving figure")
            pass
        # plt.close()
        plt.close(fig)
    else:
        # plt.close()
        # plt.close(fig)
        
        plt.show()
        pass


def getRelHat(orientation_s1, orientation_s2):
    rel_hat = np.zeros_like(orientation_s1)  # Placeholder for relative quaternions
    # Step 1: Calculate relative quaternions and Euler angles
    for t in range(orientation_s1.shape[0]):  # Iterate over the quaternions (rows)
        rel_hat[t, :] = quatmultiply(quatconj(orientation_s1[t, :]), orientation_s2[t, :])  
    return rel_hat

    
def getDistDiff_plot_ori(data, N, rel_hat, PATH, filename, if_plot=False):
    """
    Compare real orientation (data["qREF"]) vs estimated orientation (rel_hat).
    Computes angular distance and per-axis differences (x, y, z).

    Parameters:
    - data: dict containing "qREF" (Nx4 quaternion array)
    - N: number of samples
    - rel_hat: estimated quaternions (Nx4)
    - PATH: path to save plots
    - filename: filename prefix
    - if_plot: whether to plot angular distance
    """
    angular_dist = np.zeros(N)  # Placeholder for angular distances
    for t in range(N):  # N is the number of data points
        # angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t+1, :])  # Angular distance calculation
        angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t, :])  # Angular distance calculation
        
    print(np.mean(angular_dist))
    rel_eu = quaternion_to_euler(rel_hat).T
    qRef_eu = quaternion_to_euler(data["qREF"]).T

    # euler_diff = rel_eu.T - qRef_eu.T
    euler_diff = rel_eu - qRef_eu
    time_vec = np.arange(rel_eu.shape[0])  # or use your actual time vector

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot estimated Euler angles
    axs[0,0].plot(time_vec, rel_eu[:,0], label='X', alpha=0.8)
    axs[0,0].plot(time_vec, rel_eu[:,1], label='Y', alpha=0.8)
    axs[0,0].plot(time_vec, rel_eu[:,2], label='Z', alpha=0.8)
    axs[0,0].set_title('Estimated Euler angles (rel_eu)')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Angle (deg)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Plot reference Euler angles
    axs[0,1].plot(time_vec, qRef_eu[:,0], label='X', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu[:,1], label='Y', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu[:,2], label='Z', alpha=0.8)
    axs[0,1].set_title('Reference Euler angles (qRef_eu)')
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Angle (deg)')
    axs[0,1].legend()
    axs[0,1].grid(True)

    # Plot Euler differences
    axs[1,0].plot(time_vec, euler_diff[:,0], label='X', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,1], label='Y', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,2], label='Z', alpha=0.8)
    axs[1,0].set_title('Euler angle differences (rel - ref)')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Difference (deg)')
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Optional: leave last subplot empty or use for stats
    axs[1,1].axis('off')  # just empty for layout

    plt.tight_layout()
    plt.show()
    # data["angular_dist"] = angular_dist
    if if_plot is True:
        plot_angular_dist(angular_dist, PATH+filename+"_dis.png", True)
    else:
        plot_angular_dist(angular_dist, PATH+filename+"_dis.png", False )
    return angular_dist


def getDistDiff_plot_ori_random(reference_q, N, drift_q, PATH, filename, if_plot=False):
    """
    Compare real orientation (data["qREF"]) vs estimated orientation (rel_hat).
    Computes angular distance and per-axis differences (x, y, z).

    Parameters:
    - data: dict containing "qREF" (Nx4 quaternion array)
    - N: number of samples
    - rel_hat: estimated quaternions (Nx4)
    - PATH: path to save plots
    - filename: filename prefix
    - if_plot: whether to plot angular distance
    """
    # dft_eu = quaternion_to_euler(drift_q).T
    # qRef_eu = quaternion_to_euler(reference_q).T
    dft_eu = np.degrees(quaternion_to_euler(drift_q)).T
    qRef_eu = np.degrees(quaternion_to_euler(reference_q)).T
    # euler_diff = rel_eu.T - qRef_eu.T
    euler_diff = dft_eu - qRef_eu
    time_vec = np.arange(dft_eu.shape[0])  # or use your actual time vector

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot estimated Euler angles
    axs[0,0].plot(time_vec, dft_eu[:,0], label='X', alpha=0.8)
    axs[0,0].plot(time_vec, dft_eu[:,1], label='Y', alpha=0.8)
    axs[0,0].plot(time_vec, dft_eu[:,2], label='Z', alpha=0.8)
    axs[0,0].set_title('Estimated Euler angles (dft)')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Angle (deg)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Plot reference Euler angles
    axs[0,1].plot(time_vec, qRef_eu[:,0], label='X', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu[:,1], label='Y', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu[:,2], label='Z', alpha=0.8)
    axs[0,1].set_title('Reference Euler angles (q)')
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Angle (deg)')
    axs[0,1].legend()
    axs[0,1].grid(True)

    # Plot Euler differences
    axs[1,0].plot(time_vec, euler_diff[:,0], label='X', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,1], label='Y', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,2], label='Z', alpha=0.8)
    axs[1,0].set_title('Euler angle differences (rel - ref)')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Difference (deg)')
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Optional: leave last subplot empty or use for stats
    axs[1,1].axis('off')  # just empty for layout

    plt.tight_layout()
    plt.show()

    def remove_spikes(eu, threshold=280):
        eu_clean = eu.copy()
        eu_clean[np.abs(eu_clean) > threshold] = np.nan
        print(eu_clean)
        return eu_clean

    dft_eu_clean = remove_spikes(dft_eu)
    qRef_eu_clean = remove_spikes(qRef_eu)

    # Difference also ignoring spikes
    euler_diff = remove_spikes(dft_eu_clean - qRef_eu_clean)

    time_vec = np.arange(dft_eu.shape[0])  # or your actual time vector

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Estimated Euler
    axs[0,0].plot(time_vec, dft_eu_clean[:,0], label='X', alpha=0.8)
    axs[0,0].plot(time_vec, dft_eu_clean[:,1], label='Y', alpha=0.8)
    axs[0,0].plot(time_vec, dft_eu_clean[:,2], label='Z', alpha=0.8)
    axs[0,0].set_title('Estimated Euler angles (dft)')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Angle (deg)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Reference Euler
    axs[0,1].plot(time_vec, qRef_eu_clean[:,0], label='X', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu_clean[:,1], label='Y', alpha=0.8)
    axs[0,1].plot(time_vec, qRef_eu_clean[:,2], label='Z', alpha=0.8)
    axs[0,1].set_title('Reference Euler angles (q)')
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Angle (deg)')
    axs[0,1].legend()
    axs[0,1].grid(True)

    # Euler differences
    axs[1,0].plot(time_vec, euler_diff[:,0], label='X', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,1], label='Y', alpha=0.8)
    axs[1,0].plot(time_vec, euler_diff[:,2], label='Z', alpha=0.8)
    axs[1,0].set_title('Euler angle differences (rel - ref)')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Difference (deg)')
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Empty subplot
    axs[1,1].axis('off')

    plt.tight_layout()
    plt.show()


    return 1

def getRMSE_location(r1_list, r2_list, data_t, PATH, filename):
    # Calculate RMSE for location
    rmse_1 = np.sqrt(np.mean((r1_list - data_t["r1"])**2 ))
    rmse_2 = np.sqrt(np.mean((r2_list - data_t["r2"])**2 ))
    print(f"RMSE segment 1: {rmse_1}")
    print(f"RMSE segment 2: {rmse_2}")
    return rmse_1, rmse_2

# def plot_location(r_est, r_gt, PATH, filename):
#     # Plot estimated vs ground truth location
#     # location is (N, 3)
#     plt.figure()



def getDistDiff(data, N, rel_hat,  PATH, filename, if_plot=False):
    angular_dist = np.zeros(N)  # Placeholder for angular distances
    for t in range(N):  # N is the number of data points
        # angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t+1, :])  # Angular distance calculation
        angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t, :])  # Angular distance calculation
        
    # print(np.mean(angular_dist))
    
    # data["angular_dist"] = angular_dist
    if PATH is not None:
        figure_path = os.path.join(PATH, filename+"_dis.png")
        if if_plot is True:
            plot_angular_dist(angular_dist, figure_path, True)
        else:
            plot_angular_dist(angular_dist, figure_path, False )
        return angular_dist
    else:
        return angular_dist 



def getDistDiff_q1q2(q1,q2,N, if_plot=False):
    angular_dist = np.zeros(N)  # Placeholder for angular distances
    for t in range(N):  # N is the number of data points
        # angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t+1, :])  # Angular distance calculation
        angular_dist[t] = angular_distance(q1[t,:],q2[t, :])  # Angular distance calculation
        
    print(np.mean(angular_dist))

    return angular_dist

def getDistDiff_ive(data, N, rel_hat,  PATH, filename, if_plot=False):
    qREF = data["qREF"]
    print("--- Inside getDistDiff_ive ---")
    print("Input N:", N)
    print("rel_hat length:", rel_hat.shape[0])
    print("qREF length:", qREF.shape[0])
    angular_dist = np.zeros(N-1)  # Placeholder for angular distances
    for t in range(N-1):  # N is the number of data points
        angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t+1, :])  # Angular distance calculation
    
        
    print(np.mean(angular_dist))
    # data["angular_dist"] = angular_dist
    if if_plot is True:
        plot_angular_dist(angular_dist, PATH+filename+"_dis.png", True)
    else:
        plot_angular_dist(angular_dist, PATH+filename+"_dis.png", False )
        
    
    return angular_dist

def save_data(data, r_opt_1, r_opt_2, orientation_s1, orientation_s2, PATH, filename):
    print("r_opt_1,  data[r1]")
    print(r_opt_1.T, data["r1"])
    print("r_opt_2,  data[r2]")
    print(r_opt_2.T, data["r2"])
    N = orientation_s1.shape[0]
    # Initialize the results arrays
    rel_hat = np.zeros_like(orientation_s1)
    rel_hatEUL = np.zeros([N, 3])
    angular_dist = np.zeros(N - 1)

    # Step 1: Calculate relative quaternions and Euler angles
    for t in range(orientation_s1.shape[0]):  # Iterate over the quaternions (rows)
        rel_hat[t, :] = quatmultiply(quatconj(orientation_s1[t, :]), orientation_s2[t, :])  
    rel_eu = quaternion_to_euler(rel_hat)
    qRef_eu = quaternion_to_euler(data["qREF"])

    euler_diff = rel_eu.T - qRef_eu.T

    # Save the result to CSV
    df = pd.DataFrame(euler_diff, columns=["roll_diff", "pitch_diff", "yaw_diff"])
    df.to_csv(PATH+filename+".csv", index=False)


    # Step 2: Calculate angular distance
    for t in range(N):  # N is the number of data points
        angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t, :])  # Angular distance calculation
    print(np.mean(angular_dist))
    data["angular_dist"] = angular_dist
    plot_angular_dist(data, PATH+filename+".png")
    return data


def save_data_ive(data, r_opt_1, r_opt_2, orientation_s1, orientation_s2, PATH, filename):
    print("r_opt_1,  data[r1]")
    print(r_opt_1.T, data["r1"])
    print("r_opt_2,  data[r2]")
    print(r_opt_2.T, data["r2"])
    N = orientation_s1.shape[0]
    # Initialize the results arrays
    rel_hat = np.zeros_like(orientation_s1)
    rel_hatEUL = np.zeros([N, 3])
    angular_dist = np.zeros(N - 1)

    # Step 1: Calculate relative quaternions and Euler angles
    for t in range(orientation_s1.shape[0]):  # Iterate over the quaternions (rows)
        rel_hat[t, :] = quatmultiply(quatconj(orientation_s1[t, :]), orientation_s2[t, :])  
    rel_eu = quaternion_to_euler(rel_hat)
    qRef_eu = quaternion_to_euler(data["qREF"])

    euler_diff = rel_eu.T[1:, :] - qRef_eu.T[:-1, :]

    # Save the result to CSV
    df = pd.DataFrame(euler_diff, columns=["roll_diff", "pitch_diff", "yaw_diff"])
    df.to_csv(PATH+filename+".csv", index=False)


    # Step 2: Calculate angular distance
    for t in range(N-1):  # N is the number of data points
        angular_dist[t] = angular_distance(data["qREF"][t,:], rel_hat[t+1, :])  # Angular distance calculation
    print(np.mean(angular_dist))
    data["angular_dist"] = angular_dist
    plot_angular_dist(data, PATH+filename+".png")
    return data