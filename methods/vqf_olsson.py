"""VQF + Olsson joint angle estimation method."""
import numpy as np
from .shared import estimate_joint_axes, calculate_joint_angle


def run_vqf_olsson(q_prox, q_dist, acc_prox, acc_dist, gyr_prox, gyr_dist):
    """Estimate joint angle using VQF + Olsson method.

    Args:
        q_prox: Proximal segment orientation quaternions (N, 4)
        q_dist: Distal segment orientation quaternions (N, 4)
        acc_prox: Proximal accelerometer data (N, 3) or (3, N)
        acc_dist: Distal accelerometer data (N, 3) or (3, N)
        gyr_prox: Proximal gyroscope data (N, 3) or (3, N)
        gyr_dist: Distal gyroscope data (N, 3) or (3, N)

    Returns:
        (angle_deg, jhat_prox, jhat_dist): Joint angle in degrees and estimated joint axes
    """
    # Estimate joint axes using Olsson method
    jhat_prox, jhat_dist = estimate_joint_axes(acc_prox, acc_dist, gyr_prox, gyr_dist)

    # Calculate joint angle using quaternion projection
    angle_deg = calculate_joint_angle(q_prox, q_dist, jhat_prox)

    return angle_deg, jhat_prox, jhat_dist
