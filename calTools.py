import numpy as np
from constants import FS, T, CUTOFF, GN
from scipy.signal import butter, filtfilt, correlate

def lowpass_filter(
    data,             # input signal array
    cutoff=CUTOFF,    # cutoff frequency in Hz
    fs=FS,            # sampling frequency in Hz
    order=4,          # filter order
    axis_operate=0,   # axis along which to filter
):
    """Apply Butterworth lowpass filter, returns filtered signal."""
    print("Applying lowpass filter with cutoff:", cutoff, "Hz")
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=axis_operate)


def angular_acceleration_body(
    t_array,        # time values (T,)
    cycle=0.1,      # frequency scaling factor
    motion_type=3,  # 1=1D, 2=2D, 3=3D motion
):
    """Compute angular acceleration from time steps, returns (T, 3) array."""
    t_array = np.asarray(t_array)
    T = t_array.shape[0]
    omega_freq = cycle

    omega_dot = np.zeros((T, 3))

    if motion_type == 1:
        omega_dot[:, 0] = 1 * omega_freq * np.cos(omega_freq * t_array)
        # y and z stay zero

    elif motion_type == 2:
        omega_dot[:, 0] = 1.5 * omega_freq * np.cos(omega_freq * t_array)     # d/dt of sin
        omega_dot[:, 1] = -1.3 * omega_freq * np.sin(omega_freq * t_array)    # d/dt of cos

    elif motion_type == 3:
        omega_dot[:, 0] = 1.5 * omega_freq * np.cos(omega_freq * t_array)     # d/dt of sin
        omega_dot[:, 1] = -1.3 * omega_freq * np.sin(omega_freq * t_array)    # d/dt of cos
        omega_dot[:, 2] = -1.2 * omega_freq * np.sin(omega_freq * t_array)    # d/dt of cos

    else:
        raise ValueError("motion_type must be 1, 2, or 3")

    return omega_dot.T


def compute_r(
    acc1,       # accelerometer measurements sensor 1 (3, N)
    acc2,       # accelerometer measurements sensor 2 (3, N)
    gyr1,       # gyroscope measurements sensor 1 (3, N)
    gyr2,       # gyroscope measurements sensor 2 (3, N)
    q_lin_s1,   # quaternion trajectory sensor 1 (N, 4)
    q_lin_s2,   # quaternion trajectory sensor 2 (N, 4)
    N,          # number of samples
    fs=FS,      # sampling frequency in Hz
):
    """Estimate sensor-to-joint center vectors via least squares, returns (r1, r2)."""
    A_list = []
    b_list = []

    dgyr1 = approx_derivative(gyr1, fs)
    dgyr2 = approx_derivative(gyr2, fs)

    for i in range(N):
        R1 = quat2matrix(q_lin_s1[i, :])
        R2 = quat2matrix(q_lin_s2[i, :])

        C1 = crossM(gyr1[:,i]) @ crossM(gyr1[:,i]) + crossM(dgyr1[:,i])
        C2 = crossM(gyr2[:,i]) @ crossM(gyr2[:,i]) + crossM(dgyr2[:,i])

        A_t = np.hstack([R1 @ C1, -R2 @ C2])
        b_t = ((R1 @ acc1[:,i]) - (R2 @ acc2[:,i])).reshape(3,1)
        A_list.append(A_t)
        b_list.append(b_t)

    A_dense = np.vstack(A_list)
    b_dense = np.vstack(b_list)

    # TODO: add cov_link in compute_r
    r, residuals, rank, s = np.linalg.lstsq(A_dense, b_dense, rcond=None)

    r1_opt = r[:3].reshape(3,1)
    r2_opt = r[3:].reshape(3,1)

    return r1_opt, r2_opt

def alignDataE(
    qIS,  # estimated quaternion trajectory (N, 4)
    qVM,  # reference quaternion trajectory (N, 4)
):
    """Align estimated trajectory to reference using SVD, returns (qIS, qIS_adapted, qMS_est, qVI_est)."""
    qIS_mat = qIS.T
    qVM_mat = qVM.T
    N_sample = qIS_mat.shape[1]

    A = np.zeros((4, 4))
    for i in range(N_sample):
        qVM_i = qVM_mat[:, i].flatten()
        qIS_i = qIS_mat[:, i].flatten()
        A += quatL(qVM_i).T @ quatR(qIS_i)

    U, _, V_T = np.linalg.svd(A)
    V = V_T.T

    qMS_est = U[:, 0] / np.linalg.norm(U[:, 0])
    qVI_est = V[:, 0] / np.linalg.norm(V[:, 0])

    qIS_adapted = np.zeros_like(qIS)
    qSM_est = quatconj(qMS_est)

    for i in range(N_sample):
        qIS_adapted[i, :] = quatmultiply(quatmultiply(qVI_est, qIS[i, :]), qSM_est)

    return qIS, qIS_adapted, qMS_est, qVI_est

def quaternion_to_euler(q):  # quaternion array (N, 4)
    """Convert quaternions to Euler angles (roll, pitch, yaw), returns (3, N)."""
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return np.vstack([roll, pitch, yaw])

def quatinv(q):  # quaternion (4,)
    """Compute quaternion inverse, returns (4,)."""
    conj = quatconj(q)
    norm_sq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    if norm_sq == 0:
        print("Quaternion has zero norm, cannot invert.")
        return q
    return np.array(conj/norm_sq)


def quatconj(q):  # quaternion (4,)
    """Compute quaternion conjugate, returns (4,)."""
    q_conj = q.copy()
    q_conj[1:] *= -1
    return q_conj


def angular_distance(
    s,  # first quaternion (4,)
    r,  # second quaternion (4,)
):
    """Calculate angular distance between two quaternions, returns degrees."""
    quat_diff = quatmultiply(quatinv(r), s)

    if quat_diff[0] < 0:
        quat_diff = -quat_diff

    if quat_diff[0]**2 > 1:
        quat_diff = quatnormalize(quat_diff)

    angle = 2 * np.arccos(quat_diff[0])
    return angle * 180 / np.pi


def dlogdq():
    """Derivative of quaternion log map, returns (3, 4) matrix."""
    M = np.zeros((4, 3))
    M[1:4, :] = np.eye(3)
    return M.T


def dexpndn():
    """Derivative of quaternion exp map, returns (4, 3) matrix."""
    return dlogdq().T

def EXPr(v):  # rotation vector (3,)
    """Rotation matrix exponential map (Rodrigues formula), returns (3, 3)."""
    nv = np.linalg.norm(v)
    v_unit = v / nv
    vX = crossM(v_unit)
    # R = I + sin(θ)*K + (1-cos(θ))*K²  where K is skew-symmetric matrix of unit axis
    R = np.eye(3) + np.sin(nv) * vX + (1 - np.cos(nv)) * (vX @ vX)
    return R


def EXPq(v):  # rotation vector (3,)
    """Quaternion exponential map, returns unit quaternion (4,)."""
    if v.ndim == 1 and len(v) == 3:
        v = v.reshape(1, 3)

    norm_v = np.linalg.norm(v)
    v = v.flatten()
    q = np.hstack([np.cos(norm_v), (v / (norm_v + np.finfo(float).eps)) * np.sin(norm_v)])
    return q


def LOGq(q):  # unit quaternion (4,)
    """Quaternion logarithm map, returns rotation vector (3,)."""
    if q[0]**2 > 1:
        q = quatnormalize(q)

    qv = q[1:]
    norm_qv = np.linalg.norm(qv)
    # v = arccos(q0) / ||qv|| * qv
    v = (np.arccos(q[0]) / (norm_qv + np.finfo(float).eps)) * qv
    return v


def dexpnCdexpn():
    """Conjugate derivative matrix for exp map, returns (4, 4) with diag(-1,-1,-1,-1) except (0,0)=1."""
    M = -np.eye(4)
    M[0, 0] = 1
    return M


def crossM(qv):  # 3-element vector
    """Skew-symmetric cross-product matrix, returns (3, 3)."""
    # TODO: RENAME it into crossMfromVec
    qv = np.asarray(qv).ravel()

    if qv.size != 3:
        raise ValueError(f"crossM requires a 3-element vector, but got shape {qv.shape}")

    return np.array([[0, -qv[2], qv[1]],
                     [qv[2], 0, -qv[0]],
                     [-qv[1], qv[0], 0]])

def quatL(q):  # quaternion (4,) or pure vector (3,)
    """Left quaternion multiplication matrix, returns (4, 4)."""
    if q.shape[0] == 3:
        q = np.hstack(([0], q))

    q0 = q[0]
    qv = q[1:4]

    qL = np.zeros((4, 4))
    qL[0, 0] = q0
    qL[0, 1:4] = -qv
    qL[1:4, 0] = qv
    qL[1:4, 1:4] = q0 * np.eye(3) + crossM(qv)

    return qL


def quatR(q):  # quaternion (4,) or pure vector (3,)
    """Right quaternion multiplication matrix, returns (4, 4)."""
    if q.shape[0] == 3:
        q = np.concatenate(([0], q))

    q0 = q[0]
    qv = q[1:4]

    qR = np.zeros((4, 4))
    qR[0, 0] = q0
    qR[0, 1:4] = -qv
    qR[1:4, 0] = qv
    qR[1:4, 1:4] = q0 * np.eye(3) - crossM(qv)

    return qR

def integrateGyr_differentT(
    gyr,   # gyroscope measurements (N, 3)
    q_1,   # initial quaternion (4,)
    time,  # time array (N,)
):
    """Integrate gyroscope with variable time steps, returns quaternion trajectory (N, 4)."""
    orientation = np.zeros((gyr.shape[0], 4))
    orientation[0, :] = q_1

    for i in range(1, gyr.shape[0]):
        orientation[i, :] = quatmultiply(orientation[i-1, :], EXPq(((time[i]-time[i-1]) / 2) * gyr[i-1, :]))

    return orientation


def integrateGyr(
    gyr,    # gyroscope measurements (N, 3)
    q_1,    # initial quaternion (4,)
    dt=T,   # time step in seconds
):
    """Integrate gyroscope with constant time step, returns quaternion trajectory (N, 4)."""
    orientation = np.zeros((gyr.shape[0], 4))
    orientation[0, :] = q_1

    for i in range(1, gyr.shape[0]):
        orientation[i, :] = quatmultiply(orientation[i-1, :], EXPq((dt / 2) * gyr[i-1, :]))

    return orientation


def dLnk_etaG(
    R,  # rotation matrix (3, 3)
    C,  # centripetal term vector (3,)
):
    """Derivative of link constraint w.r.t. global orientation error, returns (3, 3)."""
    v = R @ C
    return crossM(v)


def dLnk(
    R,  # rotation matrix (3, 3)
    C,  # centripetal term vector (3,)
):
    """Derivative of link constraint w.r.t. local orientation error, returns (3, 3)."""
    return R @ crossM(C)


def dLnkdr_etaG(
    R,  # rotation matrix (3, 3)
    K,  # kinematic matrix (3, 3)
):
    """Derivative of link constraint w.r.t. position vector (global), returns (3, 3)."""
    return R @ K


def dLnkdr(
    R,  # rotation matrix (3, 3)
    K,  # kinematic matrix (3, 3)
):
    """Derivative of link constraint w.r.t. position vector (local), returns (3, 3)."""
    return R @ K

def dAcc(q_lin):  # linearization quaternion (4,)
    """Derivative of acceleration cost w.r.t. orientation, returns (3, 3)."""
    R = quat2matrix(q_lin)
    return crossM(R.T @ GN)


def dInit_etaG(
    q_1,     # initial quaternion (4,)
    q_lin,   # linearization quaternion (4,)
):
    """Derivative of init cost w.r.t. global orientation error, returns (3, 3)."""
    q_lin_conj = quatconj(q_lin)
    q_mult = quatmultiply(q_1, q_lin_conj)
    return dlogdq() @ quatR(q_mult) @ dexpndn()


def dInit(
    q_1,     # initial quaternion (4,)
    q_lin,   # linearization quaternion (4,)
):
    """Derivative of init cost w.r.t. local orientation error, returns (3, 3)."""
    q_1_conj = quatconj(q_1)
    q_mult = quatmultiply(q_1_conj, q_lin)
    return dlogdq() @ quatL(q_mult) @ dexpndn()


def dMotion_tp1_etaG(
    q_lin_tp1,  # linearization quaternion at t+1 (4,)
    q_lin_t,    # linearization quaternion at t (4,)
    dt=T,       # time step in seconds
):
    """Derivative of motion cost w.r.t. state at t+1 (global), returns (3, 3)."""
    return (1/dt) * dlogdq() @ quatL(quatconj(q_lin_t)) @ quatR(q_lin_tp1) @ dexpndn()


def dMotion_t_etaG(
    q_lin_tp1,  # linearization quaternion at t+1 (4,)
    q_lin_t,    # linearization quaternion at t (4,)
    dt=T,       # time step in seconds
):
    """Derivative of motion cost w.r.t. state at t (global), returns (3, 3)."""
    # q(uv) = q(vu)^c conjugate relation
    return (1/dt) * dlogdq() @ quatL(quatconj(q_lin_t)) @ quatR(q_lin_tp1) @ dexpnCdexpn() @ dexpndn()


def dMotion(
    q_lint,     # linearization quaternion at t (4,)
    q_lintm1,   # linearization quaternion at t-1 (4,)
    dt=T,       # time step in seconds
):
    """Derivative of motion cost w.r.t. state at t (local), returns (3, 3)."""
    q_lintm1_conj = quatconj(q_lintm1)
    q_mult = quatmultiply(q_lintm1_conj, q_lint)
    return (1 / dt) * dlogdq() @ quatL(q_mult) @ dexpndn()


def dMotiontm1(
    q_lint,     # linearization quaternion at t (4,)
    q_lintm1,   # linearization quaternion at t-1 (4,)
    dt=T,       # time step in seconds
):
    """Derivative of motion cost w.r.t. state at t-1 (local), returns (3, 3)."""
    q_rel = quatmultiply(quatconj(q_lintm1), q_lint)
    return (1/dt) * dlogdq() @ quatR(q_rel) @ dexpnCdexpn() @ dexpndn() 
    
    
def quatnormalize(q):  # quaternion (4,)
    """Normalize quaternion to unit length with positive scalar, returns (4,)."""
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat2matrix(q):  # quaternion (4,)
    """Convert unit quaternion to rotation matrix, returns (3, 3)."""
    q = quatnormalize(q)
    w, x, y, z = q

    R = np.array([
        [w**2 + x**2 - y**2 - z**2, 2 * (x*y - z*w),       2 * (x*z + y*w)],
        [2 * (x*y + z*w),       w**2 - x**2 + y**2 - z**2, 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),       w**2 - x**2 - y**2 + z**2]
    ])

    return R

def update_linPoints_etaG(
    n_G,       # rotation vector increments (N, 3)
    q_lin_G,   # current linearization quaternions (N, 4)
):
    """Update linearization points with global perturbation, returns (N, 4)."""
    q_lin_G_ = np.zeros_like(q_lin_G)

    for i in range(q_lin_G.shape[0]):
        q_lin_G_[i, :] = quatmultiply(EXPq(n_G[i, :] / 2), q_lin_G[i, :])

    return q_lin_G_


def update_linPoints(
    q_lin,  # current linearization quaternions (N, 4)
    n,      # rotation vector increments (N, 3)
):
    """Update linearization points with local perturbation, returns (N, 4)."""
    q_lin_ = np.zeros_like(q_lin)

    for i in range(q_lin.shape[0]):
        q_lin_[i, :] = quatmultiply(q_lin[i, :], EXPq(n[i, :] / 2))

    return q_lin_

def quatmultiply(
    q,  # first quaternion (4,)
    r,  # second quaternion (4,)
):
    """Multiply two quaternions q*r, returns (4,)."""
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r

    w = q0*r0 - q1*r1 - q2*r2 - q3*r3
    x = q0*r1 + q1*r0 + q2*r3 - q3*r2
    y = q0*r2 + q2*r0 + q3*r1 - q1*r3
    z = q0*r3 + q3*r0 + q1*r2 - q2*r1

    return np.array([w, x, y, z])


def approx_derivative(
    y,            # signal array (3, N) or (N, 3)
    fs=FS,        # sampling frequency in Hz
    lpf_dw=False, # apply lowpass filter before differentiation
):
    """Compute numerical derivative using 5-point central difference."""
    if lpf_dw:
        print("y.shape:", y.shape)
        print("apply the lowpass filter for calculating w derivatives")
        y = lowpass_filter(y.T).T
    dy = np.zeros_like(y)
    dy[:, 2:-2] = (y[:, :-4] - 8 * y[:, 1:-3] + 8 * y[:, 3:-1] - y[:, 4:]) * (fs / 12)
    return dy


def calc_acc_at_center(
    gyr,   # gyroscope measurements (3, N)
    dgyr,  # gyroscope derivative (3, N)
    acc,   # accelerometer measurements (3, N)
    r,     # position vector from sensor to center (3,)
):
    """Compute acceleration at joint center removing centripetal terms, returns (AccG, Cr)."""
    Cr = np.zeros((3, gyr.shape[1]))
    AccG = np.zeros((3, gyr.shape[1]))
    for i in range(gyr.shape[1]):
        # Centripetal + tangential acceleration: omega x (omega x r) + alpha x r
        Cr[:,i:i+1] = crossM(gyr[:,i]) @ crossM(gyr[:,i]) @ r.reshape(3,1) + crossM(dgyr[:,i]) @ r.reshape(3,1)
        AccG[:,i] = acc[:,i] - Cr[:,i]
    return AccG, Cr


def calculate_convergence_metrics(
    r_true,      # ground truth position (3,)
    r_est,       # estimated positions (N, 3)
    n_converge,  # sample index where convergence is reached
):
    """Calculate total and stable RMSE for position estimates, returns dict with metrics."""
    r_est = np.array(r_est)
    r_true = np.array(r_true).flatten()

    errors = np.linalg.norm(r_est - r_true, axis=1)
    rmse_total = np.sqrt(np.mean(errors**2))

    if n_converge < len(errors):
        stable_errors = errors[n_converge:]
        rmse_stable = np.sqrt(np.mean(stable_errors**2))
    else:
        rmse_stable = np.nan
        print("Warning: n_converge is greater than the total number of samples.")

    return {
        'rmse_total': rmse_total,
        'rmse_stable': rmse_stable,
        'final_error': errors[-1]
    }