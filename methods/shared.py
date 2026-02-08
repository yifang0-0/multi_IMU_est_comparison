"""Shared utility functions for estimation methods."""
import xml.etree.ElementTree as ET

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


def calculate_joint_angle(
    q_rel,       # relative quaternion (N, 4) in [w, x, y, z] convention
    jhat_prox,   # unit joint axis in proximal sensor frame (3,)
):
    """Calculate joint angle via swing-twist decomposition around joint axis, returns degrees."""
    angle_rad = qmt.quatProject(q_rel, jhat_prox)['projAngle']
    return np.degrees(angle_rad)


def _euler_xyz_to_rotmat(euler):
    """Convert OpenSim intrinsic XYZ Euler angles (radians) to rotation matrix."""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz



def parse_osim_calibration(
    model_path,   # path to calibrated .osim file
    joint_name,   # joint name substring to match (e.g. 'knee_r')
    prox_imu,     # proximal IMU frame name (e.g. 'femur_r_imu')
    dist_imu,     # distal IMU frame name (e.g. 'tibia_r_imu')
):
    """Parse calibrated .osim model for IMU-to-joint calibration frames, returns dict."""
    tree = ET.parse(model_path)
    root = tree.getroot()

    # Find the joint matching joint_name (CustomJoint or PinJoint)
    joint_el = None
    joint_type = None
    for tag in ('CustomJoint', 'PinJoint'):
        for jt in root.iter(tag):
            if joint_name in jt.get('name', ''):
                joint_el = jt
                joint_type = tag
                break
        if joint_el is not None:
            break
    if joint_el is None:
        raise ValueError(f"No joint matching '{joint_name}' in {model_path}")

    # Extract parent and child offset frame orientations from the joint
    parent_orient = child_orient = None
    for pof in joint_el.iter('PhysicalOffsetFrame'):
        name = pof.get('name', '')
        orient_el = pof.find('orientation')
        if orient_el is None:
            continue
        orient = np.array([float(v) for v in orient_el.text.split()])
        # Parent frame is listed first (socket_parent_frame), child second
        if parent_orient is None:
            parent_orient = orient
        else:
            child_orient = orient

    if parent_orient is None or child_orient is None:
        raise ValueError(f"Could not find parent/child offset frames in joint '{joint_name}'")

    # Extract rotation axis: CustomJoint has SpatialTransform, PinJoint rotates about Z
    if joint_type == 'PinJoint':
        rot_axis = np.array([0.0, 0.0, 1.0])
    else:
        rot_axis = np.array([1.0, 0.0, 0.0])  # default X-axis
        for ta in joint_el.iter('TransformAxis'):
            if ta.get('name') == 'rotation1':
                axis_el = ta.find('axis')
                if axis_el is not None:
                    rot_axis = np.array([float(v) for v in axis_el.text.split()])
                break

    # Find IMU PhysicalOffsetFrame orientations (top-level, not inside joint)
    imu_orients = {}
    for pof in root.iter('PhysicalOffsetFrame'):
        name = pof.get('name', '')
        if name in (prox_imu, dist_imu):
            orient_el = pof.find('orientation')
            if orient_el is not None:
                imu_orients[name] = np.array([float(v) for v in orient_el.text.split()])

    if prox_imu not in imu_orients:
        raise ValueError(f"IMU frame '{prox_imu}' not found in {model_path}")
    if dist_imu not in imu_orients:
        raise ValueError(f"IMU frame '{dist_imu}' not found in {model_path}")

    return {
        'R_prox_proxIMU': _euler_xyz_to_rotmat(imu_orients[prox_imu]),
        'R_dist_distIMU': _euler_xyz_to_rotmat(imu_orients[dist_imu]),
        'R_parent_offset': _euler_xyz_to_rotmat(parent_orient),
        'R_child_offset': _euler_xyz_to_rotmat(child_orient),
        'rot_axis': rot_axis / np.linalg.norm(rot_axis),
    }


def calculate_joint_angle_model(
    q_rel,   # relative quaternion array (N, 4) in [w, x, y, z]
    calib,   # calibration dict from parse_osim_calibration
):
    """Calculate joint angle using model calibration chain with swing-twist decomposition, returns degrees."""
    # Precompute constant frame transforms: A @ R_imu_rel @ B = R_joint
    A = calib['R_parent_offset'].T @ calib['R_prox_proxIMU']
    B = calib['R_dist_distIMU'].T @ calib['R_child_offset']

    # Apply calibration chain: (N,3,3) = (3,3) @ (N,3,3) @ (3,3)
    R_imu = qmt.quatToRotMat(q_rel)             # (N, 3, 3)
    R_joint = np.einsum('ij,njk,kl->nil', A, R_imu, B)
    q_joint = qmt.quatFromRotMat(R_joint)        # (N, 4)

    # Extract angle via swing-twist decomposition about the rotation axis
    angle_rad = qmt.quatProject(q_joint, calib['rot_axis'])['projAngle']
    return np.degrees(angle_rad)


