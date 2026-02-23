"""Direct joint angle estimation using Seel et al. method (gyro+acc fusion without orientation)."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from dfjimu import estimate_lever_arms
from .shared import olsson_estimate_hinge_joint_axes


def _skew(v):
    """Skew-symmetric matrix for cross product: skew(v) @ u = v x u."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _rodrigues_rotation(angle, axis):
    """Rotation matrix for rotation by angle (rad) about unit axis (Rodrigues formula)."""
    axis = axis / np.linalg.norm(axis)
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _forward_diff(signal, fs):
    """Forward difference derivative matching Scilab d_dt."""
    d = np.empty_like(signal)
    d[:-1] = (signal[1:] - signal[:-1]) * fs
    d[-1] = d[-2] if len(signal) > 1 else 0
    return d


def _axes_aligned(j_prox, j_dist, gyr_prox, gyr_dist, n_est, dt):
    """Parallelepiped volume indicator for axis alignment (positive = aligned)."""
    n = len(gyr_prox)

    # Split gyro into axis projection (scalar) and perpendicular component (vector)
    a_prox = gyr_prox @ j_prox                          # (N,)
    p_prox = gyr_prox - np.outer(a_prox, j_prox)        # (N, 3)
    a_dist = gyr_dist @ j_dist
    p_dist = gyr_dist - np.outer(a_dist, j_dist)

    # Collect (p_pre, p_post, meanrot) pairs at n_est-sample intervals
    vol_prox_list = []
    vol_dist_list = []
    for i in range(2 * n_est - 1, n, n_est):  # Scilab: 2*n_est:n_est:$  (1-indexed)
        i_pre = i - n_est
        # Skip if any NaN in window
        if np.any(np.isnan(p_prox[i_pre:i + 1])) or np.any(np.isnan(p_dist[i_pre:i + 1])):
            continue

        # Mean rotation around axis during interval (matching Scilab weighting)
        # Scilab: mean([a(i+1-n_est:i-1), mean(a([i-n_est,i]))])
        interior_prox = a_prox[i_pre + 1:i]          # n_est - 2 interior samples
        endpoints_mean_prox = np.mean(a_prox[[i_pre, i]])
        meanrot_prox = np.mean(np.append(interior_prox, endpoints_mean_prox))

        interior_dist = a_dist[i_pre + 1:i]
        endpoints_mean_dist = np.mean(a_dist[[i_pre, i]])
        meanrot_dist = np.mean(np.append(interior_dist, endpoints_mean_dist))

        # Correct p_post by rotating back by accumulated rotation around axis
        rot_angle_prox = meanrot_prox * dt * n_est
        rot_angle_dist = meanrot_dist * dt * n_est
        p_post_corr_prox = _rodrigues_rotation(rot_angle_prox, j_prox) @ p_prox[i]
        p_post_corr_dist = _rodrigues_rotation(rot_angle_dist, j_dist) @ p_dist[i]

        # Signed parallelepiped volume: (p_pre x p_post_corrected) . j
        vol_prox_list.append(np.cross(p_prox[i_pre], p_post_corr_prox) @ j_prox)
        vol_dist_list.append(np.cross(p_dist[i_pre], p_post_corr_dist) @ j_dist)

    if len(vol_prox_list) == 0:
        return 0.0

    vol_prox = np.array(vol_prox_list)
    vol_dist = np.array(vol_dist_list)
    denom = np.sum(np.maximum(np.abs(vol_prox), np.abs(vol_dist)) ** 2)
    if denom < 1e-30:
        return 0.0
    return np.sum(vol_prox * vol_dist) / denom


def _align_axis_signs(gyr_prox, gyr_dist, jhat_prox, jhat_dist, fs):
    """Align joint axis signs using correlation and parallelepiped volume indicator."""
    n_est = max(1, round(0.1 * fs))  # 10 at 100Hz
    dt = 1.0 / fs

    # Step 1: Ensure both axes project angular rates in the same direction (correlation check)
    w_prox = gyr_prox @ jhat_prox
    w_dist = gyr_dist @ jhat_dist
    if np.corrcoef(w_prox, w_dist)[0, 1] < 0:
        jhat_dist = -jhat_dist

    # Step 2: Align proximal axis via parallelepiped volume indicator (Scilab axes_aligned)
    indicator = _axes_aligned(jhat_prox, jhat_dist, gyr_prox, gyr_dist, n_est, dt)
    if indicator < 0:
        jhat_prox = -jhat_prox

    return jhat_prox, jhat_dist


def _gyro_angle(gyr_prox, gyr_dist, jhat_prox, jhat_dist, fs):
    """Integrate differential angular velocity around joint axis (trapezoidal), returns angle_rad."""
    alpha_dot = gyr_prox @ jhat_prox - gyr_dist @ jhat_dist
    dt = 1.0 / fs
    alpha_gyr = np.zeros(len(alpha_dot))
    for i in range(1, len(alpha_dot)):
        alpha_gyr[i] = alpha_gyr[i - 1] + 0.5 * (alpha_dot[i - 1] + alpha_dot[i]) * dt
    return alpha_gyr


def _acc_angle(
    acc_prox,     # (N, 3) proximal accelerometer (pre-filtered)
    acc_dist,     # (N, 3) distal accelerometer (pre-filtered)
    gyr_prox,     # (N, 3) proximal gyroscope
    gyr_dist,     # (N, 3) distal gyroscope
    jhat_prox,    # (3,) joint axis in proximal frame
    jhat_dist,    # (3,) joint axis in distal frame
    o_prox,       # (3,) joint position in proximal frame
    o_dist,       # (3,) joint position in distal frame
    fs,           # sampling frequency
):
    """Compute accelerometer-based angle by correcting for centripetal/tangential acc, returns radians."""
    n = len(acc_prox)

    # Gyro derivative (forward difference, matching Scilab d_dt)
    gyr_dot_prox = _forward_diff(gyr_prox, fs)
    gyr_dot_dist = _forward_diff(gyr_dist, fs)

    # Joint plane basis vectors (matching Scilab: cross([1,0,0], j) and cross(cross([1,0,0], j), j))
    ref = np.array([1.0, 0.0, 0.0])
    v1_prox = np.cross(ref, jhat_prox)
    v2_prox = np.cross(v1_prox, jhat_prox)
    v1_dist = np.cross(ref, jhat_dist)
    v2_dist = np.cross(v1_dist, jhat_dist)

    # Fallback if joint axis is nearly parallel to [1,0,0]
    if np.linalg.norm(v1_prox) < 1e-8:
        ref2 = np.array([0.0, 1.0, 0.0])
        v1_prox = np.cross(ref2, jhat_prox)
        v2_prox = np.cross(v1_prox, jhat_prox)
    if np.linalg.norm(v1_dist) < 1e-8:
        ref2 = np.array([0.0, 1.0, 0.0])
        v1_dist = np.cross(ref2, jhat_dist)
        v2_dist = np.cross(v1_dist, jhat_dist)

    alpha_acc = np.zeros(n)
    for i in range(n):
        # K = [omega_x]^2 + [omega_dot_x]  (centripetal + tangential)
        K_prox = _skew(gyr_prox[i]) @ _skew(gyr_prox[i]) + _skew(gyr_dot_prox[i])
        K_dist = _skew(gyr_dist[i]) @ _skew(gyr_dist[i]) + _skew(gyr_dot_dist[i])

        # Correct accelerations for motion artifacts
        a_corr_prox = acc_prox[i] - K_prox @ o_prox
        a_corr_dist = acc_dist[i] - K_dist @ o_dist

        # Project into joint plane: [v1_component, v2_component]
        a1_0 = a_corr_dist @ v1_dist
        a1_1 = a_corr_dist @ v2_dist
        a2_0 = a_corr_prox @ v1_prox
        a2_1 = a_corr_prox @ v2_prox

        # Solve rotation: [[-a1_1, a1_0], [a1_0, a1_1]] @ [sin, cos]' = [a2_0, a2_1]'
        denom = a1_0 ** 2 + a1_1 ** 2
        if denom > 1e-10:
            # Closed-form: angle between 2D vectors a1 and a2
            alpha_acc[i] = np.arctan2(
                a1_0 * a2_1 - a1_1 * a2_0,  # cross product (sin)
                a1_0 * a2_0 + a1_1 * a2_1,  # dot product (cos)
            )
        elif i > 0:
            alpha_acc[i] = alpha_acc[i - 1]

    return alpha_acc


def _complementary_filter(alpha_gyr, alpha_acc_filt, psi=0.01):
    """Simple complementary filter fusing gyro-integrated and acc-based angles."""
    n = len(alpha_gyr)
    alpha_cf = np.zeros(n)
    alpha_cf[0] = alpha_acc_filt[0]
    for i in range(1, n):
        alpha_cf[i] = psi * alpha_acc_filt[i] + (1 - psi) * (
            alpha_gyr[i] - alpha_gyr[i - 1] + alpha_cf[i - 1]
        )
    return alpha_cf


def run_seel(acc_prox, gyr_prox, acc_dist, gyr_dist, fs):
    """Seel direct angle estimation with complementary filter fusion."""
    # Ensure (N, 3) shape
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # 1. Joint axis estimation (Olsson)
    jhat_prox, jhat_dist = olsson_estimate_hinge_joint_axes(
        acc_prox, acc_dist, gyr_prox, gyr_dist
    )
    jhat_prox, jhat_dist = _align_axis_signs(gyr_prox, gyr_dist, jhat_prox, jhat_dist, fs)

    # 2. Joint position estimation (lever arms)
    o_prox, o_dist = estimate_lever_arms(gyr_prox, gyr_dist, acc_prox, acc_dist, fs)

    # 3. Gyro-based angle
    alpha_gyr = _gyro_angle(gyr_prox, gyr_dist, jhat_prox, jhat_dist, fs)

    # 4. Acc-based angle (pre-filter accelerometer, matching Scilab)
    n_lpfilt = max(1, round(0.02 * fs))
    win = 2 * n_lpfilt + 1
    acc_prox_filt = uniform_filter1d(acc_prox, size=win, axis=0, mode='nearest')
    acc_dist_filt = uniform_filter1d(acc_dist, size=win, axis=0, mode='nearest')
    alpha_acc = _acc_angle(
        acc_prox_filt, acc_dist_filt, gyr_prox, gyr_dist,
        jhat_prox, jhat_dist, o_prox, o_dist, fs
    )
    alpha_acc_filt = uniform_filter1d(alpha_acc, size=win, mode='nearest')

    # 5. Complementary filter fusion
    alpha_cf = _complementary_filter(alpha_gyr, alpha_acc_filt, psi=0.01)
    return np.degrees(alpha_cf), jhat_prox, jhat_dist, o_prox, o_dist
