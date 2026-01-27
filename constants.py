# constants.py - Physics and sampling constants
import numpy as np

# Physics
G = 9.82
GN = np.array([0, 0, -G])

# Sampling
FS = 100          # Hz
T = 1 / FS        # Sampling period
CUTOFF = 6.0      # Low-pass filter cutoff (Hz)
N_STILL = 40      # Stationary samples for noise estimation

# KF covariance scales (tune these for different datasets)
COV_W_SCALE = 1e-2              # Process noise
COV_LNK_SCALE = 0.35**2 * 10    # Measurement noise (link constraint)
ACC_OUTLIER_THRESHOLD = 300     # m/s² - reject samples above this
