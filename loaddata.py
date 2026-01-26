import os
import numpy as np
from scipy.io import loadmat
from os.path import exists
# NOTE: These modules are assumed to be correctly imported from your environment:
from calTools import quatmultiply, quatconj, approx_derivative, calc_acc_at_center, angular_acceleration_body
from GlobalParams import params 


# --- Utility Functions ---

def _normalize_sensor_shape(arr, name):
    """Ensures sensor arrays are in the required (Axes, Time) format."""
    if arr.ndim == 1 and arr.size == 3:
        # Array is (3,) -> Reshape to (3, 1) if it's a lever arm, or (3, N) if N=1.
        return arr.reshape(3, 1)

    if arr.ndim != 2:
        # Not a time series array we can normalize
        return arr
        
    # Check if the shape is (N, 3) (Time x Axes - MATLAB format)
    if arr.shape[0] != 3 and arr.shape[1] == 3:
        # Array is (Time, Axes) -> Transpose to (Axes, Time)
        print(f"Normalizing {name} by transposing from {arr.shape} to (3, N).")
        return arr.T
    elif arr.shape[0] == 3 and arr.shape[1] != 3:
        # Array is already (Axes, Time) (NPZ format)
        return arr
    else:
        # Unexpected shape (e.g., non-3D data or empty). Use original.
        return arr

def _generate_time_array(data, N, params):
    """Generates a time array based on N and params.T if 't' is missing."""
    if 't' in data and data['t'] is not None and (isinstance(data['t'], np.ndarray) and data['t'].size > 0):
        return data['t']
    if N > 0 and hasattr(params, 'T'):
        return np.arange(0, N) * params.T 
    return None

def load_raw_data(filename, type="matIve"):
    """
    Loads data from .mat (handling MATLAB structure format or root level variables) 
    or .npz file. Returns a dictionary of data.
    """
    if type == "matIve":
        try:
            mat_contents = loadmat(filename, struct_as_record=False, squeeze_me=True)
            data = mat_contents.get('data')
            if data is None:
                mat_dict = {k: v for k, v in mat_contents.items() if not k.startswith('__')}
                if not mat_dict:
                    print(f"Error: MAT file {filename} appears empty or corrupt.")
                    return None
                return mat_dict
            if isinstance(data, np.ndarray):
                data = data[0]
            mat_dict = {}
            for key in data.__dict__:
                value = getattr(data, key)
                if isinstance(value, np.ndarray) and value.dtype == 'O' and value.size > 0:
                    mat_dict[key] = value[0]
                else:
                    mat_dict[key] = value
            return mat_dict
        except Exception as e:
            print(f"Error loading MATLAB file {filename}: {e}")
            return None
    
    # type == "npz"
    try:
        return dict(np.load(filename, allow_pickle=True))
    except FileNotFoundError:
        return None

def extract_N_timesteps(data, N):
    """
    Truncates data arrays to N timesteps from the beginning (index 0).
    """
    extracted_data = {}
    for key, value in data.items():
        if not isinstance(value, np.ndarray) or value.size == 0:
            extracted_data[key] = value
            continue

        if value.ndim == 1 and value.shape[0] >= N:
            extracted_data[key] = value[0:N]
        elif value.ndim == 2:
            if value.shape[1] >= N:
                extracted_data[key] = value[:, 0:N]
            elif value.shape[0] >= N:
                extracted_data[key] = value[0:N, :]
            else:
                extracted_data[key] = value
        else:
            extracted_data[key] = value
            
    return extracted_data 


# --- Generic Calculation Function (Format Agnostic) ---

def _run_calculations(data, params):
    """
    Performs all signal processing and calculation steps, assuming the 
    data keys are already standardized and normalized to (Axes, Time).
    """
    r1, r2 = data['r1'], data['r2']
    acc_1, gyr_1 = data['acc_1'], data['gyr_1']
    acc_2, gyr_2 = data['acc_2'], data['gyr_2']
    N = gyr_1.shape[1]

    # --- Relative Quaternion Calculation ---
    if 'relRef' in data:
        qREF_final = data['relRef']
        print("Using direct 'relRef' as qREF.")
    elif 'qGS1_ref' in data:
        qGS1_ref, qGS2_ref = data['qGS1_ref'], data['qGS2_ref']
        qREF_final = np.array([quatmultiply(quatconj(qGS1_ref[i]), qGS2_ref[i]) for i in range(qGS1_ref.shape[0])])
        print("Calculating 'qREF' from qGS1_ref and qGS2_ref.")
    else:
        raise ValueError("Cannot calculate qREF: Missing all reference orientation sources ('relRef' and 'qGS1/qGS2').")

    # Gyro noise covariance (Now robust to shape due to normalization)
    
    # Determine the time axis for slicing (must be axis 1 if normalized to (3, N))
    gyr1_still = gyr_1[:, :params.N_still]
    gyr2_still = gyr_2[:, :params.N_still]
    
    sdNoise_gyr1 = np.std(gyr1_still, ddof=1, axis=1) # axis=1 is now safe
    sdNoise_gyr2 = np.std(gyr2_still, ddof=1, axis=1)
    
    cov_w_vector = np.concatenate((sdNoise_gyr1.flatten(), sdNoise_gyr2.flatten()))
    cov_w = np.diag(cov_w_vector)
    
    cov_i = np.eye(3) * 0.35**2 

    # Angular Acceleration Derivative
    dgyr_1 = approx_derivative(gyr_1, params.Fs)
    dgyr_2 = approx_derivative(gyr_2, params.Fs)
    
    # --- DIMENSION ALIGNMENT FIX (Final check before external call) ---
    N_calc = min(gyr_1.shape[1], dgyr_1.shape[1], gyr_2.shape[1], dgyr_2.shape[1])
    
    # Re-slice ALL time-dependent arrays to this minimum length
    data['gyr_1'] = gyr_1[:, :N_calc]
    data['dgyr_1'] = dgyr_1[:, :N_calc]
    data['acc_1'] = acc_1[:, :N_calc]
    data['gyr_2'] = gyr_2[:, :N_calc]
    data['dgyr_2'] = dgyr_2[:, :N_calc]
    data['acc_2'] = acc_2[:, :N_calc]
    
    if 'qGS1_ref' in data:
        data['qGS1_ref'] = data['qGS1_ref'][:N_calc, :]
        data['qGS2_ref'] = data['qGS2_ref'][:N_calc, :]
    
    # Re-assign inputs for calc_acc_at_center using the clipped data
    gyr_1, dgyr_1, acc_1 = data['gyr_1'], data['dgyr_1'], data['acc_1']
    gyr_2, dgyr_2, acc_2 = data['gyr_2'], data['dgyr_2'], data['acc_2']
    N = N_calc 

    # Center of Mass Acceleration
    AccG1, Cr1 = calc_acc_at_center(gyr_1, dgyr_1, acc_1, r1)
    AccG2, Cr2 = calc_acc_at_center(gyr_2, dgyr_2, acc_2, r2)
    
    # Final data update
    data.update({
        'qREF': qREF_final,  
        'cov_w': cov_w, 'cov_i': cov_i, 
        'AccG1': AccG1, 'Cr1': Cr1, 'AccG2': AccG2, 'Cr2': Cr2,
        'apxdgry1': dgyr_1, 'apxdgry2': dgyr_2,
        'dgyr1': data.get('dgyr1', dgyr_1), 
        'dgyr2': data.get('dgyr2', dgyr_2), 
    })
    
    return data, N


def _get_mat_variable(data, key_name, num=0):
    """
    Retrieves a variable, searching the root and then checking for nesting 
    under the 'sensors' key (if 'sensors' is a structure/dictionary).
    """
    # 1. Check the root level dictionary
    val = data.get(key_name)
    if val is not None:
        return val

    # 2. Check for nesting under the 'sensors' structure/key
    sensors_data = data.get('sensors')
    if isinstance(sensors_data, dict):
        # Case: sensors is a dictionary (Python convention)
        return sensors_data[num].get(key_name)
    
    if hasattr(sensors_data[num], key_name):
        print(f"Found variable '{key_name}' under 'sensors {num}' structure.")
        # Case: sensors is a generic object (scipy convention for structs)
        return getattr(sensors_data[num], key_name, None)
    
    return None # Variable not found

# --- Main File Handler ---

def data_transmat2npz(data_path, FORCE_REPROCESS=False):
    # ... (omitted setup and path determination) ...
    data_name = os.path.basename(data_path).split('.')[0]
    data = None
    needs_processing = False
    
    data_path_npz = data_path.replace('.mat', '.npz') if data_path.lower().endswith('.mat') else data_path
    data_path_mat = data_path_npz.replace('.npz', '.mat')
    
    # 1. Check for Existing Target NPZ file
    if exists(data_path_npz) and not FORCE_REPROCESS:
        print(f"Attempting to load data **{data_name}** from NPZ: {data_path_npz}")
        data = load_raw_data(data_path_npz, type="npz")
        
        if data is not None and 'qREF' in data:
            print(f"Data **{data_name}** already processed and complete.")
            return dict(data)
        
        if data is not None:
            print("NPZ file found but appears incomplete. Reprocessing required.")
            
            # --- NPZ Data Extraction (Normalization Applied) ---
            data['r1'] = _normalize_sensor_shape(-data['r1'], 'r1 (NPZ)')
            data['r2'] = _normalize_sensor_shape(-data['r2'], 'r2 (NPZ)')
            print(f"Normalized lever arms r1 and r2 to shape: {data['r1'].shape}, {data['r2'].shape}")
            data['acc_1'] = _normalize_sensor_shape(data['acc_1'], 'acc_1 (NPZ)')
            data['gyr_1'] = _normalize_sensor_shape(data['gyr_1'], 'gyr_1 (NPZ)')
            data['acc_2'] = _normalize_sensor_shape(data['acc_2'], 'acc_2 (NPZ)')
            data['gyr_2'] = _normalize_sensor_shape(data['gyr_2'], 'gyr_2 (NPZ)')
            
            if 'relRef' in data: data['relRef'] = data['relRef']
            
            data.setdefault('accjc', np.array([0])) 
            data.setdefault('raw_acc_1', data['acc_1'])
            data.setdefault('raw_gyr_1', data['gyr_1'])
            data.setdefault('raw_acc_2', data['acc_2'])
            data.setdefault('raw_gyr_2', data['gyr_2'])
            
            needs_processing = True

    # 2. Check for Raw MAT file
    if not needs_processing and exists(data_path_mat):
        print(f"Data **{data_name}** exists as MAT. Loading and processing required.")
        data = load_raw_data(data_path_mat, type="matIve") 
        
        if data is None:
            print(f"Failed to load raw data from {data_path_mat}.")
            return None
            
        # --- MAT Data Extraction (Normalization Applied) ---
        # --- VARIABLE FETCHING (Uses new robust method) ---
        sensors_matrix = _get_mat_variable(data, 'sensors')
        q_matrix = _get_mat_variable(data, 'q', 0)
        # r_12_src = _get_mat_variable(data, 'r_12')
        # r_21_src = _get_mat_variable(data, 'r_21')
        
        if sensors_matrix is None:
            raise KeyError("Fatal Error: Could not find mandatory 'sensors' data matrix.")
        # Determine r1/r2 (Mapping or pseudo value)
        if 'r_12' in data:
            data['r1'] = _normalize_sensor_shape(-data['r_12'], 'r1 (MAT)')
        elif 'r1' in data:
            data['r1'] = _normalize_sensor_shape(data['r1'], 'r1 (MAT)')
        else:
            data['r1'] = np.array([[1], [1], [1]]) 
            print("Warning: 'r_12' not found in MAT file. Using pseudo value [1, 1, 1] for r1.")
        if 'r_21' in data:
            data['r2'] = _normalize_sensor_shape(-data['r_21'], 'r2 (MAT)')
        elif 'r2' in data:
            data['r2'] = _normalize_sensor_shape(data['r2'], 'r2 (MAT)')
        else:
            data['r2'] = np.array([[1], [1], [1]]) 
            print("Warning: 'r_21' not found in MAT file. Using pseudo value [1, 1, 1] for r2.")
            
        # Sensor data mapping (Normalized to 3 x N)
        try:
            data['acc_1'] = _normalize_sensor_shape(data['acc'], 'acc (MAT)')
            data['gyr_1'] = _normalize_sensor_shape(data['gyr'], 'gyr (MAT)')
            data['acc_2'] = _normalize_sensor_shape(data['acc_2'], 'acc_2 (MAT)')
            data['gyr_2'] = _normalize_sensor_shape(data['gyr_2'], 'gyr_2 (MAT)')

            if q_matrix is not None:
                print(f"Quaternion reference 'q' found in MAT file with shape {q_matrix.shape}.")
                if q_matrix.ndim != 2:
                    raise ValueError("Quaternion data 'q' must be a 2D array.")

                if q_matrix.shape[0] == 4:
                    data['qGS1_ref'] = _get_mat_variable(data, 'q', 0).T
                    data['qGS2_ref'] = _get_mat_variable(data, 'q', 1).T
                    print("Detected 'q' (N x 4). Extracted and transfered qGS1_ref and qGS2_ref.")
                else:
                    raise ValueError(f"Quaternion data 'q' found (Shape: {q_matrix.shape}) but not N x 4 or N x 8 format.")
            else:
                print("Warning: Quaternion reference 'q' not found in MAT file.")

            if 'relRef' in data:
                data['relRef'] = data['relRef']
                print("Using direct 'relRef' from MAT file.")
            elif 'ref' in data:
                data['qGS1_ref'] = data['ref'][:, 0:4]
                data['qGS2_ref'] = data['ref'][:, 4:8]
                print("Extracted qGS1_ref and qGS2_ref from 'ref' in MAT file.")
            
        except KeyError as e:
            print(f"Fatal Error: Missing expected sensor data key: {e}. Expected root keys: 'acc', 'gyr', 'acc_2', 'gyr_2', 'ref' or 'relRef'.")
            return None

        # Set raw keys
        data.setdefault('accjc', data.get('accjc', np.array([0])))
        data['raw_acc_1'] = data['acc_1'].copy()
        data['raw_gyr_1'] = data['gyr_1'].copy()
        data['raw_acc_2'] = data['acc_2'].copy()
        data['raw_gyr_2'] = data['gyr_2'].copy()
        
        needs_processing = True
    # data['relRef'] = np.array([quatmultiply(quatconj(data["qGS1_ref"][i]), data["qGS2_ref"][i]) for i in range(data["qGS1_ref"].shape[0])])

    # 3. Final File Not Found Check
    if not needs_processing:
        print(f"Data {data_name} not found")
        return None

    # 4. Processing and Saving
    if needs_processing:
        # Get N from normalized data
        N = data['gyr_1'].shape[1] 
        print(f"Data {data_name} loaded with {N} timesteps. Starting calculations...")
        
        data, N = _run_calculations(data, params) 
        data['t'] = _generate_time_array(data, N, params)
        
        # --- Save to 'processed' Subfolder ---
        folder = os.path.dirname(data_path_npz)
        filename = os.path.basename(data_path_npz)
        processed_folder = os.path.join(folder, "processed")
        print(f"Saving processed data to: {processed_folder}")
        os.makedirs(processed_folder, exist_ok=True)
        save_path = os.path.join(processed_folder, filename)
        
        # Define the keys to save (using 'qREF' as the final required key)
        save_keys = [
            't', 'r1', 'r2', 'acc_1', 'gyr_1', 'acc_2', 'gyr_2', 'qGS1_ref', 
            'qGS2_ref', 'qREF', 'cov_w', 'cov_i', 'AccG1', 'Cr1', 'AccG2', 'Cr2', 
            'accjc', 'apxdgry1', 'apxdgry2', 'dgyr1', 'dgyr2', 
            'raw_acc_1', 'raw_gyr_1', 'raw_acc_2', 'raw_gyr_2'
        ]
        save_data = {k: data[k] for k in save_keys if k in data}

        np.savez(save_path, **save_data)
        
        print(f"Data {save_path} processed and saved as npz file")
        
        return np.load(save_path, allow_pickle=True)


# --- Main Entry Point (Remains unchanged) ---

def data_load_process(TRUNCATE, data_path, FORCE_REPROCESS=False, verbose=True):
    """
    Loads and truncates data with an optional verbose flag to control console output.
    """
    data = data_transmat2npz(data_path, FORCE_REPROCESS=FORCE_REPROCESS)
    if data is None:
        return None
    
    if not isinstance(data, dict):
        data = dict(data)

    # Note: Using .shape[1] suggests your data is stored as (3, N)
    N = data['gyr_1'].shape[1]
    
    if TRUNCATE > 0:
        if TRUNCATE < N:
            if verbose:
                print(f"Truncating from {N} to {TRUNCATE} timesteps.")
            
            data = extract_N_timesteps(data, TRUNCATE)
            N = data['gyr_1'].shape[1]
            
            if verbose:
                print(f"After truncation: **{N}** timesteps retained.")
                print("After truncation shapes:", {k: v.shape for k, v in data.items() if isinstance(v, np.ndarray)})
        else:
            if verbose:
                print(f"Requested truncation ({TRUNCATE}) >= data length ({N}). No truncation performed.")
    else:
        if verbose:
            print("No truncation performed. Current shapes:", {k: v.shape for k, v in data.items() if isinstance(v, np.ndarray)})
            
    params['N'] = N
    if verbose:
        print(f"Final data length (params['N']): **{N}**")
    
    return data