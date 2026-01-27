"""VQF + Olsson joint angle estimation method."""
import numpy as np
import qmt
from .shared import olsson_estimate_hinge_joint_axes, calculate_joint_angle


def run_vqf_olsson(acc_prox, gyr_prox, acc_dist, gyr_dist, fs):
    """Estimate joint angle using VQF + Olsson method.

    Args:
        acc_prox: Proximal accelerometer data (N, 3) or (3, N)
        gyr_prox: Proximal gyroscope data (N, 3) or (3, N)
        acc_dist: Distal accelerometer data (N, 3) or (3, N)
        gyr_dist: Distal gyroscope data (N, 3) or (3, N)
        fs: Sampling frequency in Hz

    Returns:
        (angle_deg, jhat_prox, jhat_dist, q_rel, q_prox, q_dist): Joint angle in degrees,
        estimated joint axes, relative quaternion, and segment orientations
    """
    # Estimate orientations using VQF
    q_prox = qmt.oriEstVQF(gyr_prox, acc_prox, params={'Ts': 1.0/fs})
    q_dist = qmt.oriEstVQF(gyr_dist, acc_dist, params={'Ts': 1.0/fs})

    # Estimate joint axes using Olsson method
    jhat_prox, jhat_dist = olsson_estimate_hinge_joint_axes(acc_prox, acc_dist, gyr_prox, gyr_dist)

    # Calculate relative quaternion and joint angle
    q_rel = qmt.qmult(qmt.qinv(q_prox), q_dist)
    angle_deg = calculate_joint_angle(q_rel, jhat_prox)

    return angle_deg, jhat_prox, jhat_dist, q_rel, q_prox, q_dist


def run_vqf_olsson_heading_corrected(acc_prox, gyr_prox, acc_dist, gyr_dist, fs):
    """Estimate joint angle using VQF + Olsson + heading drift correction.

    Builds on run_vqf_olsson by applying qmt.headingCorrection to remove yaw drift.

    Args:
        acc_prox: Proximal accelerometer data (N, 3) or (3, N)
        gyr_prox: Proximal gyroscope data (N, 3) or (3, N)
        acc_dist: Distal accelerometer data (N, 3) or (3, N)
        gyr_dist: Distal gyroscope data (N, 3) or (3, N)
        fs: Sampling frequency in Hz

    Returns:
        angle_deg: Joint angle in degrees, shape (N,)
    """
    Ts = 1.0 / fs

    # Get base VQF+Olsson results
    _, jhat_prox, jhat_dist, _, q_prox, q_dist = run_vqf_olsson(
        acc_prox, gyr_prox, acc_dist, gyr_dist, fs
    )

    # Align sensor frames to joint axis (z-axis = joint axis)
    q_align_prox = qmt.quatFrom2Axes(z=jhat_prox, x=acc_prox[0], exactAxis='z')
    q_align_dist = qmt.quatFrom2Axes(z=jhat_dist, x=acc_dist[0], exactAxis='z')
    q_seg_prox = qmt.qmult(q_prox, q_align_prox)
    q_seg_dist = qmt.qmult(q_dist, q_align_dist)

    # Apply heading correction to remove yaw drift
    t = qmt.timeVec(N=q_seg_prox.shape[0], Ts=Ts)
    out = qmt.headingCorrection(
        gyr1=gyr_prox, gyr2=gyr_dist,
        quat1=q_seg_prox, quat2=q_seg_dist,
        t=t, joint=[0, 0, 1], jointInfo={},
        estSettings={'constraint': 'euler_1d'}
    )

    # Calculate relative orientation and extract angle
    q_rel = qmt.qrel(q_seg_prox, out[0])
    angle_rad = np.unwrap(qmt.eulerAngles(q_rel, axes='zyx')[:, 0])

    return np.degrees(angle_rad)
