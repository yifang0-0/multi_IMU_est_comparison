import numpy as np
from constants import FS, T, CUTOFF, GN
from scipy.signal import butter, filtfilt, correlate

def lowpass_filter(data, cutoff=CUTOFF, fs=FS, order=4, axis_operate=0):
    print("Applying lowpass filter with cutoff:", cutoff, "Hz")
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=axis_operate)


def angular_acceleration_body(t_array, cycle=0.1, motion_type=3):
    """
    Computes angular acceleration (time derivative of angular velocity)
    over a vector of time steps.

    Parameters:
    - t_array: 1D numpy array of time values (shape: [T])
    - cycle: Frequency scaling factor
    - motion_type: 1, 2, or 3 (1D, 2D, or 3D motion)

    Returns:
    - omega_dot: numpy array of shape (T, 3)
    """
    t_array = np.asarray(t_array)
    T = t_array.shape[0]
    omega_freq =  cycle

    omega_dot = np.zeros((T, 3))  # Initialize output

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


def compute_r(acc1, acc2, gyr1, gyr2, q_lin_s1, q_lin_s2, N, fs=FS):
    A_list = []
    b_list = []

    dgyr1 = approx_derivative(gyr1, fs)
    dgyr2 = approx_derivative(gyr2, fs)
    
    for i in range(N):
        R1 = quat2matrix(q_lin_s1[i, :])
        R2 = quat2matrix(q_lin_s2[i, :])
    

        
        C1 = crossM(gyr1[:,i]) @ crossM(gyr1[:,i]) + crossM(dgyr1[:,i])
        C2 = crossM(gyr2[:,i]) @ crossM(gyr2[:,i]) + crossM(dgyr2[:,i])
        
        A_t = np.hstack([R1 @ C1, -R2 @ C2])  # Concatenate horizontally
        b_t = ((R1 @ acc1[:,i]) - (R2 @ acc2[:,i])).reshape(3,1)  # Compute right-hand side
        A_list.append(A_t)
        b_list.append(b_t)

 # Convert the lists to a sparse matrix for A and b
    A_dense = np.vstack(A_list)  # Stack A_t vertically (concatenate rows)
    b_dense = np.vstack(b_list)  # Stack b_t vertically (concatenate rows)
    
    # TODO: add cov_link in compute_r
    # beta_hat, _, _, _ = lstsq(F * np.sqrt(W.diagonal())[:, None], y * np.sqrt(W.diagonal()))
    # Solve the least squares problem: A @ r = b
    r, residuals, rank, s = np.linalg.lstsq(A_dense, b_dense, rcond=None) 
    
    # Extract r1 and r2
    r1_opt = r[:3].reshape(3,1)  # First p elements
    r2_opt = r[3:].reshape(3,1)  # Remaining elements

    return r1_opt, r2_opt

def alignDataE(qIS, qVM):
    # print("Inside alignDataE function"
    #       )
    # print("qIS.shape, qVM.shape", qIS.shape, qVM.shape)
    """
    Performs misalignment correction between a drifting estimated trajectory (qIS) 
    and a reference trajectory (qVM) using SVD.
    
    Args:
        qIS (np.ndarray): Estimated quaternion trajectory (N x 4, [w, x, y, z]).
        qVM (np.ndarray): Reference quaternion trajectory (N x 4, [w, x, y, z]).

    Returns:
        tuple: (qIS_raw, qIS_adapted, qMS_est, qVI_est)
    """
    # 1. Transpose data for 4 x N matrix operations (as in MATLAB)
    # The arrays must be the same length, which is handled by data_load_process.
    qIS_mat = qIS.T  # 4 x N
    qVM_mat = qVM.T  # 4 x N
    N_sample = qIS_mat.shape[1]
    
    # 2. Build the summation matrix A (4 x 4)
    # A = sum(quatL(qVM(:,i))' * quatR(qIS(:,i)))
    A = np.zeros((4, 4))
    
    for i in range(N_sample):
        # print("Processing sample ", i+1, " of ", N_sample)
        # Extract individual quaternion columns (4 x 1)
        qVM_i = qVM_mat[:, i].flatten()
        qIS_i = qIS_mat[:, i].flatten()
        # print(A.shape, "A.shape"
        #       "A: ", A  )
        # Calculate matrix A contribution (using canonical quatL and quatR matrices)
        A += quatL(qVM_i).T @ quatR(qIS_i)

    # 3. Define misalignment quaternions using SVD
    # [U, S, V] = svd(A); qMS_est = U(:,1); qVI_est = V(:,1);
    U, _, V_T = np.linalg.svd(A)
    V = V_T.T # V is transpose of V_T

    # The misalignment quaternions are the first columns of U and V
    # qMS_est is the misalignment (Sensor -> Body frame) correction
    # qVI_est is the initial alignment (Vicon -> Inertial frame) correction
    qMS_est = U[:, 0] # 4-element vector
    qVI_est = V[:, 0] # 4-element vector

    # Ensure the misalignment quaternions are correctly normalized and reshaped (1 x 4)
    qMS_est = qMS_est / np.linalg.norm(qMS_est)
    qVI_est = qVI_est / np.linalg.norm(qVI_est)
    # print("qMS_est.shape", qMS_est.shape
    #       )
    # print("qVI_est.shape", qVI_est.shape)
    # 4. Apply the Alignment to the estimated data (qIS)
    qIS_adapted = np.zeros_like(qIS)

    qIV_est = quatconj(qVI_est) # 1 x 4
    qSM_est = quatconj(qMS_est) # 1 x 4
    
    for i in range(N_sample):
    # q_temp = quatmultiply(qVI_est_conj, qVM)
    # qIS_adapted = quatmultiply(q_temp, qMS_est)

    # Step 1: q_temp[i] = qVI_est_conj * qVM[i]
        # Multiplies the static misalignment quaternion by the reference quaternion row
        # qVM[i, :] is the i-th quaternion (4,)
        # q_temp[i, :] = quatmultiply(qVI_est_conj, qVM[i, :])

        # Step 2: qIS_adapted[i] = q_temp[i] * qMS_est
        # Multiplies the intermediate result by the second static misalignment quaternion
        qIS_adapted[i, :] = quatmultiply(quatmultiply(qVI_est, qIS[i, :]), qSM_est)    # The MATLAB function returns the raw qIS input for some reason, so we follow that structure:
    # return qIS, qIS_adapted, qMS_est, qVI_est
    return qIS, qIS_adapted, qMS_est, qVI_est

# def alignDataE(qIS, qVM):
#     """
#     Python version of MATLAB function allignDataE(qIS, qVM).

#     Inputs:
#         qIS: 4×N measured drifting sensor quaternion (system I)
#         qVM: 4×N reference mount quaternion (system V → mount)

#     Outputs:
#         qIS_adapted: corrected prediction using estimated misalignments
#         qMS_est, qVI_est: estimated correction quaternions
#     """
#     print("qIS.shape, qVM.shape", qIS.shape, qVM.shape)
#     qIS = qIS.T  # Transpose to make it 4xN
#     qVM = qVM.T  # Transpose to make it 4xN
#     N = qIS.shape[1]
#     print("N=", N)
#     A = np.zeros((4, 4))

#     # -----------------------------
#     # Construct A = sum( L(qVM_i)' * R(qIS_i) )
#     # -----------------------------
#     for i in range(N):
#         A += quatL(qVM[:, i]).T @ quatR(qIS[:, i])

#     # -----------------------------
#     # SVD decomposition
#     # -----------------------------
#     U, S, Vt = np.linalg.svd(A)
#     V = Vt.T

#     qMS_est = quatnormalize(U[:, 0])
#     qVI_est = quatnormalize(V[:, 0])

#     print("qMS_est = ", qMS_est)
#     print("qVI_est = ", qVI_est)

#     # -----------------------------
#     # Compute adapted qIS (predicted)
#     # -----------------------------
#     qIS_adapted = np.zeros_like(qIS)

#     for i in range(N):
#         # qIS_adapted = qVI* ⊗ qVM ⊗ qMS
#         # print(qIS_adapted[:, i].shape, qVI_est.shape, qVM[:, i].shape, qMS_est.shape)
#         qIS_adapted[:, i] = quatmultiply(
#             quatmultiply(quatconj(qVI_est), qVM[:, i]),
#             qMS_est
#         )

#     return qIS_adapted.T, qMS_est, qVI_est


def quaternion_to_euler(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    # Roll (phi)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Pitch (theta)
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    
    # Yaw (psi)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return np.vstack([roll, pitch, yaw])

def quatinv(q):
    conj = quatconj(q)
    norm_sq = q[0]**2  + q[1]**2 + q[2]**2 + q[3]**2
    if norm_sq == 0:
        print("Quaternion has zero norm, cannot invert.")
        return q
    return np.array(conj/norm_sq)


def quatconj(q):
    q_conj = q.copy()
    q_conj[1:] *= -1
    
    return q_conj



def angular_distance(s, r):
    """
    Calculate the angular distance between two quaternions s and r.
    
    Args:
    - s (np.ndarray): A quaternion [w, x, y, z] representing the first quaternion.
    - r (np.ndarray): A quaternion [w, x, y, z] representing the second quaternion.
    
    Returns:
    - float: The angular distance in degrees.
    """
    
    # Convert the quaternions to Rotation objects
    quat_diff = quatmultiply(quatinv(r),s)
    
    if quat_diff[0]<0:
        quat_diff = -quat_diff

    # Get the angle from the quaternion difference
    if quat_diff[0]**2>1:
            # print("Quaternion norm is greater than 1, normalizing.")
            quat_diff = quatnormalize(quat_diff)
            # return 0
    angle = 2 * np.arccos(quat_diff[0])
    
    # Convert the angle from radians to degrees
    ang_d = angle*180/np.pi
    
    # Return the angular distance
    return ang_d



def dlogdq():
    # returns a 3x4 matrix
    M = np.zeros((4, 3))
    M[1:4, :] = np.eye(3)  # Set rows 2 to 4 to be the identity matrix
    M = M.T  # Transpose the matrix to make it 3x4
    return M

def dexpndn():
    """
    Returns the 4x3 matrix, which is the transpose of the matrix from dlogdq.
    """
    M = dlogdq().T  # Transpose the result of dlogdq() and return it
    
    return M

def EXPr(v):
    """
    Computes the direction cosine matrix (DCM) R corresponding to a small
    orientation deviation vector v ∈ ℝ³ using the exponential map.
    
    Parameters:
        v (np.ndarray): 3-element rotation vector
    
    Returns:
        R (np.ndarray): 3x3 rotation matrix
    """
    nv = np.linalg.norm(v)
    # if nv < 1e-8:
    #     # When nv ~ 0, sin(nv)/nv ≈ 1 and (1 - cos(nv))/nv² ≈ 0.5
    #     return np.eye(3) + crossM(v) + 0.5 * crossM(v) @ crossM(v)
    v_unit = v / nv
    vX = crossM(v_unit)
    R = np.eye(3) + np.sin(nv) * vX + (1 - np.cos(nv)) * (vX @ vX)
    return R


def EXPq(v):
    """
    Exponential of a quaternion: EXPq(n/2) for quaternion n.
    
    Parameters:
        n: np.ndarray - Input quaternion (size [4,])
    
    Returns:
        np.ndarray - Exponential of quaternion n (size [4,])
    """
    if v.ndim == 1 and len(v) == 3:
        v = v.reshape(1, 3)  # Convert to row vector if it is a column vector

    norm_v = np.linalg.norm(v)
    v=v.flatten()
    q = np.hstack([np.cos(norm_v), (v / (norm_v + np.finfo(float).eps)) * np.sin(norm_v)])  # Using np.finfo(float).eps for numerical stability
    return q

def LOGq(q):
    # Extract the vector part (qv)
    if q[0]**2>1:
        # print("Quaternion norm is greater than 1, normalizing.")
        q = quatnormalize(q)
    
    qv = q[1:]  # q[1:], i.e., qx, qy, qz
    # Compute the logarithm using the formula
    norm_qv = np.linalg.norm(qv)  # norm of the vector part
    
    v = (np.arccos(q[0]) / (norm_qv + np.finfo(float).eps)) * qv  # formula with small epsilon to avoid division by zero
    # print(np.arccos(q[0]),norm_qv,  np.finfo(float).eps,q, v)
    return v


def dexpnCdexpn():
    """
    Returns a 4x4 matrix with -I except for the (1,1) element which is 1.
    
    Returns:
        np.ndarray - 4x4 matrix
    """
    M = -np.eye(4)
    M[0, 0] = 1
    return M

def crossM(qv):
    # TODO: RENAME it into crossMfromVec
    """
    Returns the cross-product matrix of the vector part of quaternion q.
    
    Parameters:
        q: make sure the input is always a vector part of quaternion
        
    Returns:
        np.ndarray - The cross-product matrix for the vector part of q.
    """
    # if qv.ndim == 2 and qv.shape[1] == 3 and qv.shape[0] == 1:
    #     qv = qv[0, :]  # Convert to 1D array if it's a row vector
    # if qv.ndim == 2 and qv.shape[1] == 1 and qv.shape[0] == 3:
    #     qv = qv[:, 0]  # Convert to 1D array if it's a row vector
    # 强制展平为一维数组，无论输入是 (3,), (1,3) 还是 (3,1)
    # 这能彻底解决 "setting an array element with a sequence" 的问题
    qv = np.asarray(qv).ravel()
    
    if qv.size != 3:
        raise ValueError(f"crossM requires a 3-element vector, but got shape {qv.shape}")
    # Cross product matrix of vector part qv
    return np.array([[0, -qv[2], qv[1]],
                     [qv[2], 0, -qv[0]],
                     [-qv[1], qv[0], 0]])

def quatL(q):
    """
    Calculates the left quaternion multiplication matrix of quaternion q.
    Returns a 4x4 matrix.
    
    Parameters:
        q: np.ndarray - Quaternion (1x4 or 4x1)
    
    Returns:
        np.ndarray - The left multiplication matrix (4x4)
    """
    # If q is a 3-element vector, treat it as a quaternion with real part 0
    if q.shape[0] == 3:
        q = np.hstack(([0], q))  # q is now a 4-element quaternion

    q0 = q[0]  # Real part of quaternion
    qv = q[1:4]  # Vector part of quaternion

    # Initialize the 4x4 matrix
    qL = np.zeros((4, 4))

    # Fill the 4x4 matrix based on the formula
    qL[0, 0] = q0
    qL[0, 1:4] = -qv
    qL[1:4, 0] = qv
    qL[1:4, 1:4] = q0 * np.eye(3) + crossM(qv)

    return qL

def quatR(q):
    """
    Calculates the left quaternion multiplication matrix of quaternion q.
    The input quaternion q is assumed to be a 4-element vector [q0, q1, q2, q3].
    The resulting matrix is a 4x4 matrix.

    Args:
        q (np.ndarray): 4x1 or 1x4 array representing a quaternion [q0, q1, q2, q3]

    Returns:
        np.ndarray: 4x4 matrix representing the left multiplication of quaternion q.
    """
    # Ensure q is a 4-element quaternion
    if q.shape[0] == 3:
        q = np.concatenate(([0], q))  # When a vector is given, treat as quaternion with scalar part = 0

    # Extract quaternion components
    q0 = q[0]
    qv = q[1:4]

    # Initialize the quaternion multiplication matrix
    qR = np.zeros((4, 4))

    # Fill in the quaternion multiplication matrix
    qR[0, 0] = q0
    qR[0, 1:4] = -qv
    qR[1:4, 0] = qv
    qR[1:4, 1:4] = q0 * np.eye(3) - crossM(qv)

    return qR

def integrateGyr_differentT(gyr, q_1, time):
    """
    Integrates gyroscope measurements to estimate orientation (quaternion).

    Args:
        data: A numpy array of shape (N, 3) containing the gyroscope measurements (gyr).
        q_1: Initial orientation quaternion.
        params: The global parameters object containing 'T' (time step).

    Returns:
        orientation: The estimated orientations (quaternions) after integration.
    """
    # Extract global parameters from the params object
    # T = params["T"]
    # print("the time step between two samples are T: ", T)
    # Initialize the orientation array
    orientation = np.zeros((gyr.shape[0], 4))
    
    # Set the initial orientation
    orientation[0, :] = q_1
    
    # Loop over the gyro gyr and perform quaternion integration
    for i in range(1, gyr.shape[0]):
        # Compute the quaternion for the current time step
        orientation[i, :] = quatmultiply(orientation[i-1, :], EXPq(((time[i]-time[i-1]) / 2) * gyr[i-1, :]))
        # print(i+1)
        # print(np.round(orientation[i, :],4), "orientation")
        # print(np.round(gyr[i, :],4), "gyr")
    
    return orientation



def integrateGyr(gyr, q_1, dt=T):
    """
    Integrates gyroscope measurements to estimate orientation (quaternion).

    Args:
        gyr: A numpy array of shape (N, 3) containing the gyroscope measurements.
        q_1: Initial orientation quaternion.
        dt: Time step (defaults to T from constants).

    Returns:
        orientation: The estimated orientations (quaternions) after integration.
    """
    orientation = np.zeros((gyr.shape[0], 4))
    orientation[0, :] = q_1

    for i in range(1, gyr.shape[0]):
        orientation[i, :] = quatmultiply(orientation[i-1, :], EXPq((dt / 2) * gyr[i-1, :]))

    return orientation



def dLnk_etaG(R, C):
    v = R @ C
    # print("v=r@c", v.shape)
    return crossM(v)



def dLnk(R, C):
    """
    Returns the derivative of the acceleration-based link between two sensors.

    Parameters:
        R: np.ndarray - 3x3 rotation matrix
        C: np.ndarray - 3-element vector
    
    Returns:
        np.ndarray - 3x3 matrix
    """
    return R @ crossM(C)


def dLnkdr_etaG(R, K):
    """
    Returns the derivative of the acceleration-based link between two sensors
    with respect to the sensor-joint center position vector.

    Parameters:
        K: np.ndarray - 3x3 matrix
        R: np.ndarray - 3x3 rotation matrix
        n: np.ndarray - 3-element vector
    
    Returns:
        np.ndarray - 3x3 matrix
    """
    return R @ K

def dLnkdr(R, K):
    """
    Returns the derivative of the acceleration-based link between two sensors
    with respect to the sensor-joint center position vector.

    Parameters:
        K: np.ndarray - 3x3 matrix
        R: np.ndarray - 3x3 rotation matrix
        n: np.ndarray - 3-element vector
    
    Returns:
        np.ndarray - 3x3 matrix
    """
    return R @ K

def dAcc(q_lin):
    """
    Returns the derivative of the costAcc with respect to the state at time step 't'.

    Parameters:
        q_lin: np.ndarray - 4-element quaternion

    Returns:
        np.ndarray - 3x3 derivative matrix
    """
    R = quat2matrix(q_lin)
    return crossM(R.T @ GN)


def dInit_etaG(q_1, q_lin):
    """
    Returns the derivative of the costInit with respect to the state at time step 1.

    Parameters:
        q_1: np.ndarray - Initial quaternion (4-element vector)
        q_lin: np.ndarray - Linearized quaternion (4-element vector)

    Returns:
        np.ndarray - Derivative matrix (3x3 or appropriate shape based on context)
    """
    # Define the quaternion operations
    q_lin_conj = quatconj(q_lin)  # Conjugate of q_1
    q_mult = quatmultiply(q_1, q_lin_conj)  # Multiply q_1 conjugate with q_lin
    # Calculate the derivative
    der = dlogdq() @ quatR(q_mult) @ dexpndn()  # Matrix multiplication
    
    return der


def dInit(q_1, q_lin):
    """
    Returns the derivative of the costInit with respect to the state at time step 1.

    Parameters:
        q_1: np.ndarray - Initial quaternion (4-element vector)
        q_lin: np.ndarray - Linearized quaternion (4-element vector)

    Returns:
        np.ndarray - Derivative matrix (3x3 or appropriate shape based on context)
    """
    # Define the quaternion operations
    q_1_conj = quatconj(q_1)  # Conjugate of q_1
    q_mult = quatmultiply(q_1_conj, q_lin)  # Multiply q_1 conjugate with q_lin
    # Calculate the derivative
    der = dlogdq() @ quatL(q_mult) @ dexpndn()  # Matrix multiplication
    
    return der




# def quatR(q):
#     """
#     Computes the right quaternion multiplication matrix for quaternion q.
    
#     Parameters:
#         q: np.ndarray - 4-element quaternion [w, x, y, z]
    
#     Returns:
#         np.ndarray - 4x4 right multiplication matrix
#     """
#     # If input is a 3-element vector, assume it represents a pure quaternion [0, x, y, z]
#     if len(q) == 3:
#         q = np.hstack(([0], q))

#     q0 = q[0]
#     qv = np.array(q[1:])  # Extract vector part

#     qR = np.zeros((4, 4))
#     qR[0, 0] = q0
#     qR[0, 1:] = -qv
#     qR[1:, 0] = qv
#     qR[1:, 1:] = q0 * np.eye(3) - crossM(qv)  # Compute 3x3 block

#     return qR


def dMotion_tp1_etaG(q_lin_tp1, q_lin_t, dt=T):
    """
    Returns the derivative of the costMotion with respect to the state at timestep 't'.
    tutorial 4.14 b
    """
    return (1/dt) * dlogdq() @ quatL(quatconj(q_lin_t)) @ quatR(q_lin_tp1) @ dexpndn()


def dMotion_t_etaG(q_lin_tp1, q_lin_t, dt=T):
    """
    Computes the derivative of costMotion with respect to state at timestamp 't'.
    tutorial 4,14 c
    conjugate stating q(uv) = q(vu)^c
    """
    return (1/dt) * dlogdq() @ quatL(quatconj(q_lin_t)) @ quatR(q_lin_tp1) @ dexpnCdexpn() @ dexpndn()


def dMotion(q_lint, q_lintm1, dt=T):
    """
    Returns the derivative of the costMotion with respect to the state at timestep 't'.

    Parameters:
        q_lint: np.ndarray - Linearized quaternion at time step t (4-element vector)
        q_lintm1: np.ndarray - Linearized quaternion at time step t-1 (4-element vector)
        dt: float - Time duration (defaults to T from constants)

    Returns:
        np.ndarray - Derivative matrix (3x3)
    """
    q_lintm1_conj = quatconj(q_lintm1)
    q_mult = quatmultiply(q_lintm1_conj, q_lint)
    return (1 / dt) * dlogdq() @ quatL(q_mult) @ dexpndn()


def dMotiontm1(q_lint, q_lintm1, dt=T):
    """
    Computes the derivative of costMotion with respect to state at timestamp 't-1'.

    Parameters:
        q_lint: np.ndarray - Quaternion at time 't'
        q_lintm1: np.ndarray - Quaternion at time 't-1'
        dt: float - Time duration (defaults to T from constants)

    Returns:
        np.ndarray - Computed derivative (3x3)
    """
    q_rel = quatmultiply(quatconj(q_lintm1), q_lint)
    return (1/dt) * dlogdq() @ quatR(q_rel) @ dexpnCdexpn() @ dexpndn() 
    
    
def quatnormalize(q):
    """
    Normalizes a quaternion to unit length.
    
    Parameters:
        q: np.ndarray - Quaternion [w, x, y, z]
    
    Returns:
        np.ndarray - Normalized quaternion
    """
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat2matrix(q):
    """
    Converts a unit quaternion into a 3x3 rotation matrix.
    
    Parameters:
        q: np.ndarray - 4-element quaternion [w, x, y, z]
    
    Returns:
        np.ndarray - 3x3 rotation matrix
    """
    q = quatnormalize(q)  # Normalize the quaternion

    w, x, y, z = q  # Extract components

    # Construct the rotation matrix
    R = np.array([
        [w**2 + x**2 - y**2 - z**2, 2 * (x*y - z*w),       2 * (x*z + y*w)],
        [2 * (x*y + z*w),       w**2 - x**2 + y**2 - z**2, 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),       w**2 - x**2 - y**2 + z**2]
    ])
    
    return R

def update_linPoints_etaG(n_G, q_lin_G):
    """
    Update the linearization points using the rotation vectors 'n'.
    
    Parameters:
    q_lin (numpy.ndarray): Array of quaternion values to be updated.
    n (numpy.ndarray): Array of rotation vectors used for updating linearization points.

    Returns:
    numpy.ndarray: Updated linearization points.
    """
    q_lin_G_ = np.zeros_like(q_lin_G)  # Initialize the output matrix

    for i in range(q_lin_G.shape[0]):
        # Apply quaternion multiplication with the exponential map of n
    
        q_lin_G_[i, :] = quatmultiply( EXPq(n_G[i, :] / 2), q_lin_G[i, :])

    return q_lin_G_


def update_linPoints(q_lin, n):
    """
    Update the linearization points using the rotation vectors 'n'.
    
    Parameters:
    q_lin (numpy.ndarray): Array of quaternion values to be updated.
    n (numpy.ndarray): Array of rotation vectors used for updating linearization points.

    Returns:
    numpy.ndarray: Updated linearization points.
    """
    q_lin_ = np.zeros_like(q_lin)  # Initialize the output matrix

    for i in range(q_lin.shape[0]):
        # Apply quaternion multiplication with the exponential map of n
        q_lin_[i, :] = quatmultiply(q_lin[i, :], EXPq(n[i, :] / 2))

    return q_lin_

def quatmultiply(q, r):
    """
    Multiply two quaternions.

    Parameters:
    q1 (numpy.ndarray): First quaternion.
    q2 (numpy.ndarray): Second quaternion.

    Returns:
    numpy.ndarray: Resulting quaternion after multiplication.
    """
    
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r

    w = q0*r0 - q1*r1 - q2*r2 - q3*r3
    x = q0*r1 + q1*r0 + q2*r3 - q3*r2
    y = q0*r2 + q2*r0 + q3*r1 - q1*r3
    z = q0*r3 + q3*r0 + q1*r2 - q2*r1

    # return quatnormalize(np.array([w, x, y, z]))
    return np.array([w, x, y, z])



def approx_derivative(y, fs=FS, lpf_dw=False):
    """Compute numerical derivative using 5-point central difference."""
    if lpf_dw:
        print("y.shape:", y.shape)
        print("apply the lowpass filter for calculating w derivatives")
        y = lowpass_filter(y.T).T
    dy = np.zeros_like(y)
    dy[:, 2:-2] = (y[:, :-4] - 8 * y[:, 1:-3] + 8 * y[:, 3:-1] - y[:, 4:]) * (fs / 12)
    return dy

def calc_acc_at_center(gyr, dgyr, acc, r):
    Cr = np.zeros((3,gyr.shape[1]))
    AccG = np.zeros((3,gyr.shape[1]))
    for i in range(gyr.shape[1]):
        Cr[:,i:i+1] = crossM(gyr[:,i]) @ crossM(gyr[:,i]) @ r.reshape(3,1) + crossM(dgyr[:,i]) @ r.reshape(3,1)
        AccG[:,i] = acc[:,i] - Cr[:,i]
    return AccG, Cr

import numpy as np


def calculate_convergence_metrics(r_true, r_est, n_converge):
    """
    Calculates RMSE for the entire signal and the RMSE after convergence.
    
    Parameters:
    r_est (np.array): Estimated positions/orientations, shape (N, 3)
    r_true (np.array): Ground truth position/orientation, shape (3,) or (1, 3)
    n_converge (int): The sample index where convergence is reached
    
    Returns:
    dict: A dictionary containing total RMSE and stable RMSE
    """
    r_est = np.array(r_est)
    r_true = np.array(r_true).flatten() # Ensure ground truth is 1D for calculation
    
    # 1. Calculate per-sample Euclidean errors (m)
    # This is the distance between estimate and truth at each time step
    # dimension of errors: (N,)
    errors = np.linalg.norm(r_est - r_true, axis=1)
    
    # 2. Total RMSE (Root Mean Square Error for all samples)
    rmse_total = np.sqrt(np.mean(errors**2))
    
    # 3. Stable RMSE (Root Mean Square Error after convergence point)
    if n_converge < len(errors):
        stable_errors = errors[n_converge:]
        rmse_stable = np.sqrt(np.mean(stable_errors**2))
    else:
        rmse_stable = np.nan
        print("Warning: n_converge is greater than the total number of samples.")

    return {
        'rmse_total': rmse_total,
        'rmse_stable': rmse_stable,
        'final_error': errors[-1] # The error at the very last sample
    }