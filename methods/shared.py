"""Shared utility functions for estimation methods."""
import numpy as np
import pandas as pd
import qmt


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


def olsson_estimate_hinge_joint_axes(acc_femur, acc_tibia, gyr_femur, gyr_tibia):
    """Estimate joint axes using Olsson method. Returns flattened axis vectors."""
    jhat_femur, jhat_tibia = qmt.jointAxisEstHingeOlsson(
        acc_femur, acc_tibia, gyr_femur, gyr_tibia
    )
    return jhat_femur.flatten(), jhat_tibia.flatten()


def calculate_joint_angle(q_rel, jhat_prox):
    """Calculate joint angle from relative quaternion and joint axis.

    Uses qmt.quatProject for swing-twist decomposition to extract
    the rotation angle around the specified joint axis.

    Args:
        q_rel: Relative quaternion (N, 4) in [w, x, y, z] convention
        jhat_prox: Unit joint axis in proximal sensor frame (3,)
    """
    angle_rad = qmt.quatProject(q_rel, jhat_prox)['projAngle']
    return np.degrees(np.unwrap(angle_rad))
