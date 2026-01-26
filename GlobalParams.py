from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class GlobalParams:
    Fs: float  # Sampling frequency
    T: float   # Sampling period
    g: float   # Gravitational acceleration
    N: int  # Number of time steps
    q1: np.ndarray  # Initial orientation quaternion
    q2: np.ndarray  # Initial orientation quaternion
    P: np.ndarray  # Initial covariance matrix (optional)
    N_still:int # Number of motion-less time steps
    gn: np.ndarray  # Gravitational vector in 3D space
    cov_lnk: np.ndarray  # Covariance matrix for the link measurements
    cov_w:np.array
    cov_i:np.ndarray
    CUTOFF: float  # CUTOFF frequency for low-pass filter,
    if_cheat_w: bool   # Flag to indicate if we are using a cheat for the covariance of the angular velocity
    if_cheat_dw: bool   # Flag to indicate if we are using a cheat for the covariance of the angular velocity
    if_cheat_a: bool 
    run_dynamic_update: bool  
    run_measurement_update: bool 
    if_lpf_dw: bool
    iteration: int # Default value for the number of iterations
    def __getitem__(self, key):  # Allow dictionary-style access
        return getattr(self, key)

    def __setitem__(self, key, value):  # Allow setting values like a dictionary
        setattr(self, key, value)

# Now you can create `params` and use both dot notation and dictionary-style access:
params = GlobalParams(
    # Fs=50, 
    # T=1/50, 
    Fs = 100,
    T= 1/100,
    # Fs = 60,
    # T= 1/60,
    # Fs = 10,
    # T=1/10,
    g=9.82, 
    N=-1,
    q1=np.array([1, 0, 0, 0]),
    q2=np.array([1,0,0,0]),
    CUTOFF=6.0,
    # q1=np.array([ 0.98969679 , 0.10126467, -0.10100298, -0.00664243]),
    # q1=np.array([ 0.59756247, -0.62729786, -0.3834229,  -0.32000526]),

    # q2=np.array([0.63738937, 0.76724055, 0.04643119, 0.0540451 ]),
    # q2=np.array([ 0.52734721, -0.54869942, -0.49359617, -0.42094737]),

    # q2=np.array([ 0.9902681,0.0354158, 0.0017708, 0.1345799 ]),
    
    P=np.eye(3),  # Example covariance matrix,
    N_still=40,
    cov_w = np.eye(6)*1,
    gn=np.array([0, 0, -9.82]),
    cov_lnk=np.eye(3)*1,
    # cov_lnk=np.eye(3),
    iteration=100,
    # cov_i=np.eye(3) * 10,  # Example covariance matrix for inertial measurements
    cov_i = np.eye(3)*0.35*0.35,
    if_cheat_w=False,  # Default value for the cheat flag
    if_cheat_dw=False,  # Default value for the cheat flag
    if_cheat_a=False,  # Default value for the cheat flag
    if_lpf_dw = False,
    run_dynamic_update = True,
    run_measurement_update = True,
    
    
)


    
'''
    # TODO: MAKE SURE the shape things are same for all the variables
    # r1 has the shape of:  (3,) and will be copied as is
    # r2 has the shape of:  (3,) and will be copied as is
    # acc has the shape of:  (3, 3007) and will be truncated
    # gyr has the shape of:  (3, 3007) and will be truncated
    # acc_2 has the shape of:  (3, 3007) and will be truncated
    # gyr_2 has the shape of:  (3, 3007) and will be truncated
    # qGS1_ref has the shape of:  (3007, 4) and will be truncated
    # qGS2_ref has the shape of:  (3007, 4) and will be truncated
    # qREF has the shape of:  (3007, 4) and will be truncated
    # cov_w has the shape of:  (6, 6) and will be truncated
    # cov_i has the shape of:  (3, 3) and will be truncated
    # C1 has the shape of:  (3, 3007) and will be truncated
    # D1 has the shape of:  (3, 3007) and will be truncated
    # AccG2 has the shape of:  (3, 3007) and will be truncated
    # Cr2 has the shape of:  (3, 3007) and will be truncated
    # angular_dist has the shape of:  (3006,) and will be copied as is
    
    # Store results in a NumPy variable
'''
