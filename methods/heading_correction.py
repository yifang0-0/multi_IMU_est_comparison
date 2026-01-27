"""Heading correction method for joint angle estimation using VQF + Olsson + qmt.headingCorrection."""

import numpy as np
import qmt


def run_heading_correction(gyr_prox, acc_prox, gyr_dist, acc_dist, fs):
    """Estimate joint angle with VQF + Olsson + heading drift correction.

    Args:
        gyr_prox: Proximal segment gyroscope data, shape (N, 3) in rad/s
        acc_prox: Proximal segment accelerometer data, shape (N, 3) in m/s^2
        gyr_dist: Distal segment gyroscope data, shape (N, 3) in rad/s
        acc_dist: Distal segment accelerometer data, shape (N, 3) in m/s^2
        fs: Sampling frequency in Hz

    Returns:
        angle_deg: Joint angle in degrees, shape (N,)
    """
    Ts = 1.0 / fs

    # VQF orientation estimation
    q_prox = qmt.oriEstVQF(gyr_prox, acc_prox, params={'Ts': Ts})
    q_dist = qmt.oriEstVQF(gyr_dist, acc_dist, params={'Ts': Ts})

    # Olsson hinge joint axis estimation
    j_prox, j_dist = qmt.jointAxisEstHingeOlsson(acc_prox, acc_dist, gyr_prox, gyr_dist)
    j_prox, j_dist = j_prox.squeeze(), j_dist.squeeze()

    # Align sensor frames to joint axis (z-axis = joint axis)
    q_align_prox = qmt.quatFrom2Axes(z=j_prox, x=acc_prox[0], exactAxis='z')
    q_align_dist = qmt.quatFrom2Axes(z=j_dist, x=acc_dist[0], exactAxis='z')
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
