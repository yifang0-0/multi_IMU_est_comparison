"""Estimation methods for joint angle calculation."""
from .vqf_olsson import run_vqf_olsson, run_vqf_olsson_heading_corrected
from .kf_gframe import run_kf_gframe, run_kf_gframe_all_variants
from .rnno import run_rnno, compute_rnno_orientation, run_rnno_all_variants
from .seel import run_seel
from .axis import (
    estimate_joint_axis, register_axis_method, get_axis_method, list_axis_methods,
    AXIS_METHODS, OPENSIM_JOINT_AXES
)

__all__ = [
    # VQF methods
    'run_vqf_olsson', 'run_vqf_olsson_heading_corrected',
    # KF methods
    'run_kf_gframe', 'run_kf_gframe_all_variants',
    # RNNO methods
    'run_rnno', 'compute_rnno_orientation', 'run_rnno_all_variants',
    # Seel direct methods
    'run_seel',
    # Axis estimation API
    'estimate_joint_axis', 'register_axis_method', 'get_axis_method', 'list_axis_methods',
    'AXIS_METHODS', 'OPENSIM_JOINT_AXES',
]
