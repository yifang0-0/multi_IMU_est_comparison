import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import FS, T as DT
from calTools import integrateGyr, quatmultiply, quatconj, update_linPoints_etaG, EXPq, LOGq, quatR, quatL, quat2matrix, crossM, approx_derivative, calc_acc_at_center, compute_r
from Jacob import calc_jac2, calcJac_Link, calcJac_Link_r, calcJac_Link_etaG, calc_jac2_etaG
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares


def process_orientation_KF_Gframe(
    data,
    q1_init=None,
    cov_w=None,
    cov_lnk=None,
    run_dynamic_update=True,
    run_measurement_update=True,
    use_raw_gyro=False,
):
    """
    Process orientation using Extended Kalman Filter with gravity frame constraints.

    Args:
        data: Dictionary containing sensor data (gyr_1, gyr_2, acc_1, acc_2, r1, r2).
        q1_init: Initial orientation quaternion (default: [1, 0, 0, 0]).
        cov_w: Process noise covariance (6x6). Default: np.eye(6) * 1e-2.
        cov_lnk: Measurement noise covariance (3x3). Default: np.eye(3) * 0.35**2 * 10.
        run_dynamic_update: Whether to run prediction step (default: True).
        run_measurement_update: Whether to run measurement update step (default: True).
        use_raw_gyro: Use raw gyro data instead of filtered (default: False).

    Returns:
        orientation_s1, orientation_s2, P_list: Estimated orientations and covariance history.
    """
    # Defaults
    if q1_init is None:
        q1_init = np.array([1, 0, 0, 0])
    if cov_w is None:
        cov_w = np.eye(6) * 1e-2
    if cov_lnk is None:
        cov_lnk = np.eye(3) * 0.35**2 * 10

    T = DT
    gyr_1 = data['gyr_1']
    N = gyr_1.shape[1]
    gyr_2 = data['gyr_2']
    acc_1 = data['acc_1']
    acc_2 = data['acc_2']

    print("T in KF_Gframe:", T)
    print("N in KF_Gframe:", N)
    print("cov_w in KF_Gframe:", cov_w)
    print("cov_lnk in KF_Gframe:", cov_lnk)

    dgyr_1 = approx_derivative(gyr_1, FS)
    dgyr_2 = approx_derivative(gyr_2, FS)

    # Initialize linearization points for both sensors
    q_lin_s1_t = integrateGyr(gyr_1.T, q1_init)
    q_lin_s2_t = integrateGyr(gyr_2.T, q1_init)
    # r_s1 = np.array([1,1,1])
    # r_s2 = np.array([1,1,1])
    r_s1 = data['r1']
    r_s2 = data['r2']
    
    # print(f'Base, known r_1: {r_s1}')
    # print(f'Base, known r_2: {r_s2}')
    

    
    # Q = np.eye(6)*0.02
    Q = np.eye(6)*cov_w[0][0]

    # Pq_init = np.eye(6) * 0.35**2 # initial covariance matrix for the orientation
    Pq_init = np.eye(6) # initial covariance matrix for the orientation
    
    # R = np.eye(3) * 2 * (0.076**2)
    R = np.eye(3) * 2 * cov_lnk[0][0]

    # R = np.eye(3) *cov_lnk[0][0]# Measurement noise covariance matrix for the link measurements
    # R= np.eye(3) * 2*(0.076^2 # Measurement noise covariance matrix for the link measurements
    # R = np.eye(3) * 1e-7 # Measurement noise covariance matrix for the link measurements
    
    G = T * np.eye(6) # TODO: WHAT SHOULD BE g
    
    # Compute initial strapdown integration
    orientation_s1 = np.zeros((N, 4))
    orientation_s2 = np.zeros((N, 4))
    orientation_s1[0] = q1_init
    orientation_s2[0] = q1_init

    P_list = []
    P_list.append(Pq_init.copy())

    num_rejected = 0
    # Initialize linearization
    q_lin_s1_t = q1_init.copy()
    q_lin_s2_t = q1_init.copy()
    P_local = np.zeros((6, 6))
    P_local[0:6,0:6] = Pq_init.copy()
    x0 = np.zeros((6, 1))  # Initial state vector (orientation and position)
    # x0[0:3] = q_lin_s1_t
    # x0[3:6] = q_lin_s2_t
    
    x_local = x0.copy()
    
    for t in range(1, N):
        gyr_1_t = gyr_1[:, t-1:t]
        gyr_2_t = gyr_2[:, t-1:t]
        acc_1_t = acc_1[:, t:t+1]
        acc_2_t = acc_2[:, t:t+1]
        dgyr_1_t = dgyr_1[:, t:t+1]
        dgyr_2_t = dgyr_2[:, t:t+1]
        # TODO: SHAPE OF THE DGYR_2
    

        eta = np.zeros((2, 3)) 
    
        x_local[0:3,0] = eta[0,0:3] 
        x_local[3:6,0] = eta[1,0:3] 
        # print("before q_lin_s1_t", q_lin_s1_t)
        

        # acceleration abnormal detector
        # !!!!!!!!!! attention: depends on the dataset!
        if (np.any(acc_1_t > 300) or np.any(acc_2_t > 300)):
            run_acc_inlimit = False
            num_rejected+=1
            print(f"detected uncommon {t} acc_1_t, acc_2_t: ", acc_1_t, acc_2_t)
        else:
            run_acc_inlimit = True


        # Time Update
        # -------------------------
        # Dynamic Update (Prediction)
        # -------------------------
        

        # if run_dynamic_update and run_acc_inlimit:

        if run_dynamic_update:

            # Time Update
            F = np.eye(6)
            # R is the same 
            q_lin_s1_t = quatmultiply(q_lin_s1_t, EXPq(T/2 * gyr_1_t))
            q_lin_s2_t = quatmultiply(q_lin_s2_t, EXPq(T/2 * gyr_2_t))


            if use_raw_gyro:
                gyr_1_t = data['raw_gyr_1'][:, t:t+1]
                gyr_2_t = data['raw_gyr_2'][:, t:t+1]
            else:
                gyr_1_t = gyr_1[:, t:t+1]
                gyr_2_t = gyr_2[:, t:t+1]
            AccG1_t, Cr1_t = calc_acc_at_center(gyr_1_t, dgyr_1_t, acc_1_t, r_s1)
            AccG2_t, Cr2_t = calc_acc_at_center(gyr_2_t, dgyr_2_t, acc_2_t, r_s2)

            G = np.zeros((6, 6))
            G[:3, :3] = T * quat2matrix(q_lin_s1_t)
            G[3:6, 3:6] = T * quat2matrix(q_lin_s2_t)
            P_local[0:6, 0:6] = F @ P_local[0:6, 0:6] @ F.T + G @ Q[0:6,0:6] @ G.T

        else:
            AccG1_t, Cr1_t = calc_acc_at_center(gyr_1_t, dgyr_1_t, acc_1_t, r_s1)
            AccG2_t, Cr2_t = calc_acc_at_center(gyr_2_t, dgyr_2_t, acc_2_t, r_s2)
        # -------------------------
        # Measurement Update
        # -------------------------
        if run_measurement_update and run_acc_inlimit:
        # if run_measurement_update:

            # Measurement Update
            H = np.zeros((3, 6))
        # --- 增加调试保护 ---
            try:
                # 先计算中间变量
                vec1 = quat2matrix(q_lin_s1_t) @ AccG1_t
                vec2 = quat2matrix(q_lin_s2_t) @ AccG2_t
                
                # 在执行 crossM 前检查
                H[0:3, 0:3] = crossM(vec1)
                H[0:3, 3:6] = -crossM(vec2)
                
            except Exception as e:
                print(f"\n!!! Error detected at sample t={t} !!!")
                print(f"Error Type: {type(e).__name__}: {e}")
                print("N: ", N)
                # 打印关键变量的 Shape
                print(f"DEBUG SHAPES:")
                print(f"  AccG1_t: {AccG1_t.shape if hasattr(AccG1_t, 'shape') else 'no shape'}")
                print(f"  q_lin_s1_t: {q_lin_s1_t.shape}")
                print(f"  vec1 (result of @): {vec1.shape if 'vec1' in locals() else 'N/A'}")
                
                # 检查是否包含 NaN
                print(f"DEBUG VALUES (NaN check):")
                print(f"  AccG1_t has NaN: {np.isnan(AccG1_t).any() if hasattr(AccG1_t, 'shape') else 'N/A'}")
                print(f"  AccG1_t content: {AccG1_t}")
                
                # 打印原始传感器数据，看是不是数据源断了
                print(f"RAW DATA at t={t}:")
                print(f"  acc_1_t: {acc_1_t.flatten()}")
                print(f"  gyr_1_t: {gyr_1_t.flatten()}")
                
                # 抛出异常终止程序，方便查看打印结果
                raise e

            # e = (quat2matrix(q_lin_s1_t) @ Cr1_t ) - (quat2matrix(q_lin_s1_t) @ Cr2_t)
            e = +(quat2matrix(q_lin_s1_t) @ AccG1_t ) - (quat2matrix(q_lin_s2_t) @ AccG2_t)
            
            
            # P_local[0:6, 0:6] = Pq_local
            # P_local[6:12, 6:12] = Pr_local
            S = H @ P_local @ H.T + R
            K = (P_local @ H.T) @ np.linalg.inv(S)
            x_local = x_local + K @ e
            eta[0,0:3] = x_local[0:3,0]
            eta[1,0:3] = x_local[3:6,0]

            
            # print("after q_lin_s1_t", q_lin_s1_t)
            
            # print("EXPq(eta[0,0:3])",EXPq(eta[0,0:3]))
            q_lin_s1_t = quatmultiply(EXPq(eta[0,0:3]/2), q_lin_s1_t)
            q_lin_s2_t = quatmultiply(EXPq(eta[1,0:3]/2), q_lin_s2_t)
            
            # P_local = P_local - K @ S @ K.T
            P_local = P_local - K @ H @ P_local
        #
        P_list.append(P_local.copy())
        # Store results
        orientation_s1[t] = q_lin_s1_t
        orientation_s2[t] = q_lin_s2_t
    
    return orientation_s1, orientation_s2, P_list
    
