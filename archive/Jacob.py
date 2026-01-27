import numpy as np
from scipy.sparse import lil_matrix,csc_matrix
from scipy.spatial.transform import Rotation as R
from calTools import dMotion, dInit, dAcc, dMotiontm1, dLnk, quat2matrix,approx_derivative, crossM,dLnkdr, dLnk_etaG, dInit_etaG, dMotion_t_etaG,dMotion_tp1_etaG


def calc_jac2( row_num, col_num, n, q_lin, q_1, cov_i, cov_w, cov_a, N):
    """
    Calculation of Jacobian matrix block for one sensor (No link).
    
    Parameters:
        M, N: int - Dimensions of the Jacobian matrix
        n: int - Unused parameter (could be removed if unnecessary)
        q_lin: np.ndarray - Linearized quaternions over time
        q_1: np.ndarray - Initial quaternion
        cov_i, cov_w, cov_a: np.ndarray - Covariance matrices
        dInit: function - Function for initial condition derivative
        dMotiontm1, dMotion: function - Functions for motion derivatives
    
    Returns:
        scipy.sparse.lil_matrix - Sparse Jacobian matrix
    """
    J = lil_matrix((row_num, col_num), dtype=np.float64)
    
    for t in range(N):
        # Initial condition
        if t == 0:
            J[0:3, 0:3] = np.linalg.inv(cov_i) ** 0.5 @ dInit(q_1, q_lin[0]) # Initial error
        
        # Dynamics error
        if t > 0:
            # q_t and q_t+1 (though it looks like q_t+1 and q_t in the code, but it starts from t=1 so it is q_t and q_t+1)
            pos_wt1_r = 3 + (t - 1) * 3
            pos_wt1_c = (t - 1) * 3
            
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c:pos_wt1_c+3] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotiontm1(q_lin[t], q_lin[t-1])
            )
            # print(t)
     
            
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c+3:pos_wt1_c+6] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion(q_lin[t], q_lin[t-1])
            )
    

    return csc_matrix(J) # use together with J = np.zeros((row_num, col_num))

def calc_jac2_etaG( row_num, col_num, n, q_lin, q_1, cov_i, cov_w, cov_a, N):
    """
    Calculation of Jacobian matrix block for one sensor (No link).
    
    Parameters:
        M, N: int - Dimensions of the Jacobian matrix
        n: int - Unused parameter (could be removed if unnecessary)
        q_lin: np.ndarray - Linearized quaternions over time
        q_1: np.ndarray - Initial quaternion
        cov_i, cov_w, cov_a: np.ndarray - Covariance matrices
        dInit: function - Function for initial condition derivative
        dMotiontm1, dMotion: function - Functions for motion derivatives
    
    Returns:
        scipy.sparse.lil_matrix - Sparse Jacobian matrix
    """
    # J = lil_matrix((row_num, col_num))  # Initialize sparse matrix
    # J = np.zeros((row_num, col_num))  # Initialize sparse matrix
    J = lil_matrix((row_num, col_num), dtype=np.float64)
    
    for t in range(N):
        # Initial condition
        if t == 0:
            J[0:3, 0:3] = np.linalg.inv(cov_i) ** 0.5 @ dInit_etaG(q_1, q_lin[0]) # Initial error
        
        # Dynamics error de{omega}/deta{i}
        if t > 0:
            pos_wt1_r = 3 + (t - 1) * 3
            pos_wt1_c = (t - 1) * 3
            
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c:pos_wt1_c+3] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion_t_etaG(q_lin[t], q_lin[t-1])
            )
            # t,t+1
            # print(t)
    
            
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c+3:pos_wt1_c+6] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion_tp1_etaG(q_lin[t], q_lin[t-1])
            )
    

    return csc_matrix(J) # use together with J = np.zeros((row_num, col_num))


def calcJac_Link(m, n, q_lin_s1, q_lin_s2, AccG1, AccG2, cov_lnk, N):
    """
    Calculation of Jacobian matrix block for the other sensor link part.
    """
    # Initialize sparse matrix
    J = lil_matrix((m,n))
    # J = np.zeros((m,n))
    

    for t in range(N):
        # Convert quaternions to rotation matrices
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])
        # Rn2 = R.from_quat(q_lin_s2[t, :]).as_matrix()

        pos_link_r = t * 3  # Row indices for this time step
        pos_link_c = t * 3  # Column indices for this time step

        # Sensor 1 (S1)
        J[pos_link_r:pos_link_r + 3, pos_link_c:pos_link_c + 3] = (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnk(Rn1, AccG1[:, t]))

        # Sensor 2 (S2), N positions further
        J[pos_link_r:pos_link_r + 3, (3 * N) + pos_link_c:(3 * N) + pos_link_c + 3] = (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnk(Rn2, AccG2[:, t]))

    return csc_matrix(J)


def calcJac_Link_etaG(m, n, q_lin_s1, q_lin_s2, AccG1, AccG2, cov_lnk, N):
    """
    Calculation of Jacobian matrix block for the other sensor link part.
    """
    ## TODO: check delink/deta
    # # Initialize sparse matrix
    J = lil_matrix((m,n))
    # # J = np.zeros((m,n))
    

    for t in range(N):
        # Convert quaternions to rotation matrices
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])
        # Rn2 = R.from_quat(q_lin_s2[t, :]).as_matrix()

        pos_link_r = t * 3  # Row indices for this time step
        pos_link_c = t * 3  # Column indices for this time step

        # Sensor 1 (S1)
        J[pos_link_r:pos_link_r + 3, pos_link_c:pos_link_c + 3] = (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnk_etaG(Rn1, AccG1[:, t]))

        # Sensor 2 (S2), N positions further
        J[pos_link_r:pos_link_r + 3, (3 * N) + pos_link_c:(3 * N) + pos_link_c + 3] = (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnk_etaG(Rn2, AccG2[:, t]))

    return J



def calcJac_Link_r(m, n, q_lin_s1, q_lin_s2, gyr_1, gyr_2, cov_lnk, Fs, N):
    """
    Calculation of Jacobian matrix block for the other sensor link part.
    """
    # Initialize sparse matrix
    J = lil_matrix((m,n))
    # J = np.zeros((m,n))

    dgyr_1 = approx_derivative(gyr_1, Fs)
    dgyr_2 = approx_derivative(gyr_2, Fs)
    

    # TODO: finish the jaccobian build
    for t in range(N):
        # Convert quaternions to rotation matrices
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])
        
        C1 = crossM(gyr_1[:,t]) @ crossM(gyr_1[:,t]) + crossM(dgyr_1[:,t])
        C2 = crossM(gyr_2[:,t]) @ crossM(gyr_2[:,t]) + crossM(dgyr_2[:,t])

        pos_link_row = t * 3  # Row indices for this time step
        pos_link_col = 0  # Column indices for this time step

        # r1 
        J[pos_link_row:pos_link_row + 3, pos_link_col:pos_link_col + 3] = (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnkdr(Rn1, C1))

        # r2 
        J[pos_link_row:pos_link_row + 3, pos_link_col+3:pos_link_col +6] = (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnkdr(Rn2, C2))

    return J

'''

def calcJacR(K1, K2, q_lin_s1, q_lin_s2, n1, n2, cov_r, N):
    """
    Calculation of Jacobian matrix block for the other sensor link part.
    """
    # Initialize sparse matrix
    J = lil_matrix((3 * N, 6))
    
    for t in range(N):
        # Convert quaternions to rotation matrices
        Rn1 = R.from_quat(q_lin_s1[t, :]).as_matrix()
        Rn2 = R.from_quat(q_lin_s2[t, :]).as_matrix()

        row = t * 3  # Row indices for this time step
        
        # For Sensor 1 (R1)
        J[row:row + 3, 0:3] = (0.05 ** -0.5) * (-dLnkdr(K1[:, :, t], Rn1, n1[t, :]))
        
        # For Sensor 2 (R2)
        J[row:row + 3, 3:6] = (0.05 ** -0.5) * dLnkdr(K2[:, :, t], Rn2, n2[t, :])

    return J



def calc_jac(M, N, n, q_lin, q_1, cov_i, cov_w, cov_a):
    """
    Calculation of the Jacobian matrix block for one sensor (No link).
    """
    # Initialize sparse matrix
    J = lil_matrix((M, N))
    
    # Loop over the time steps
    for t in range(N):
        # Initial step (first row)
        if t == 0:
            J[0:3, 0:3] = (cov_i ** -0.5) * dInit(q_1, q_lin[0, :])
        
        # Accelerometer cost (e_acc -> n(t))
        pos_acc_r = 4 + (t - 1) * 6
        pos_acc_c = 1 + (t - 1) * 3
        J[pos_acc_r:pos_acc_r+3, pos_acc_c:pos_acc_c+3] = (cov_a ** -0.5) * dAcc(q_lin[t, :])

        # Dynamics (t > 1)
        if t > 0:
            # For t-1 (ew -> n(t-1))
            pos_wt1_r = 7 + (t - 2) * 6
            pos_wt1_c = 1 + (t - 2) * 3
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c:pos_wt1_c+3] = (cov_w ** -0.5) * dMotiontm1(q_lin[t, :], q_lin[t - 1, :])

            # For t (ew -> n(t))
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c+3:pos_wt1_c+6] = (cov_w ** -0.5) * dMotion(q_lin[t, :], q_lin[t - 1, :])

    return J

'''
