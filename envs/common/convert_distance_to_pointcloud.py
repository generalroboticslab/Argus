import urdfpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load the URDF file
# urdf_path = "/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/assets/urdf/argus/argus_dof20.urdf"
urdf_path = "/home/grl/repo/vrobot_env_exp/assets/urdf/argus/argus_dof20.urdf"
robot = urdfpy.URDF.load(urdf_path)

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z])

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (x, y, z, w)"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qx, qy, qz, qw])

def apply_robot_orientation_transform(joint_origins, joint_directions, joint_quaternions, robot_orientation_matrix, robot_position=None):
    """
    Transform joint origins, directions, and quaternions based on robot's global orientation
    
    Args:
        joint_origins: List of joint positions in robot's local frame - shape: (num_joints, 3)
        joint_directions: List of joint Z-directions in robot's local frame - shape: (num_joints, 3)  
        joint_quaternions: List of joint quaternions in robot's local frame - shape: (num_joints, 4)
        robot_orientation_matrix: 3x3 or 4x4 robot orientation matrix in world frame
        robot_position: Robot position in world frame (optional) - shape: (3,)
    
    Returns:
        tuple: (transformed_origins, transformed_directions, transformed_quaternions)
    """
    
    # Ensure robot_orientation_matrix is 3x3
    if robot_orientation_matrix.shape == (4, 4):
        # Extract 3x3 rotation matrix from 4x4 homogeneous matrix
        robot_rotation_matrix = robot_orientation_matrix[:3, :3]
        if robot_position is None:
            robot_position = robot_orientation_matrix[:3, 3]  # Extract translation
    elif robot_orientation_matrix.shape == (3, 3):
        robot_rotation_matrix = robot_orientation_matrix
    else:
        raise ValueError(f"Expected 3x3 or 4x4 matrix, got {robot_orientation_matrix.shape}")
    
    # Default robot position to origin if not provided
    if robot_position is None:
        robot_position = np.array([0.0, 0.0, 0.0])
    
    robot_position = np.array(robot_position).flatten()[:3]  # Ensure it's a 3D vector
    
    print(f"Applying robot transformation:")
    print(f"Robot position: {robot_position}")
    print(f"Robot rotation matrix shape: {robot_rotation_matrix.shape}")
    
    # Convert arrays for easier processing
    joint_origins = np.array(joint_origins)
    joint_directions = np.array(joint_directions)
    joint_quaternions = np.array(joint_quaternions)
    
    # Transform joint origins (positions)
    # New_position = Robot_position + Robot_rotation @ Local_position
    transformed_origins = []
    for origin in joint_origins:
        transformed_origin = robot_position + robot_rotation_matrix @ origin
        transformed_origins.append(transformed_origin)
    
    # Transform joint directions (vectors)
    # New_direction = Robot_rotation @ Local_direction
    transformed_directions = []
    for direction in joint_directions:
        transformed_direction = robot_rotation_matrix @ direction
        # Normalize to ensure unit vector
        transformed_direction = transformed_direction / np.linalg.norm(transformed_direction)
        transformed_directions.append(transformed_direction)
    
    # Transform joint quaternions
    # Convert robot rotation matrix to quaternion
    robot_rotation = R.from_matrix(robot_rotation_matrix)
    robot_quaternion = robot_rotation.as_quat()  # Returns [x, y, z, w]
    
    transformed_quaternions = []
    for quat in joint_quaternions:
        # Convert joint quaternion to rotation object
        joint_rotation = R.from_quat(quat)  # Expects [x, y, z, w]
        
        # Compose rotations: robot_rotation * joint_rotation
        combined_rotation = robot_rotation * joint_rotation
        
        # Convert back to quaternion
        transformed_quat = combined_rotation.as_quat()  # Returns [x, y, z, w]
        transformed_quaternions.append(transformed_quat)
    
    print(f"Transformed {len(transformed_origins)} joint origins")
    print(f"Transformed {len(transformed_directions)} joint directions") 
    print(f"Transformed {len(transformed_quaternions)} joint quaternions")
    
    return np.array(transformed_origins), np.array(transformed_directions), np.array(transformed_quaternions)

def get_joint_orientations(robot, joint_angles):
    """
    Get joint orientations given joint angles.
    
    Args:
        robot: urdfpy.URDF object
        joint_angles: dict mapping joint names to angles
    
    Returns:
        dict: mapping joint names to orientation information
    """
    
    joint_orientations = {}
    
    # Build kinematic chain manually
    link_transforms = {}
    link_transforms['base_link'] = np.eye(4)  # Base at origin
    
    # Process joints to compute orientations
    processed_links = {'base_link'}
    remaining_joints = list(robot.joints)
    
    while remaining_joints:
        progress_made = False
        
        for joint in remaining_joints[:]:
            parent_link = joint.parent
            child_link = joint.child
            
            if parent_link in processed_links:
                parent_transform = link_transforms[parent_link]
                
                # Get joint's local transform
                if joint.origin is not None:
                    joint_transform = joint.origin.copy()
                else:
                    joint_transform = np.eye(4)
                
                # Apply joint motion
                if joint.joint_type != 'fixed' and joint.name in joint_angles:
                    joint_angle = joint_angles[joint.name]
                    
                    if joint.joint_type == 'revolute':
                        # Apply rotation about joint axis
                        if hasattr(joint, 'axis') and joint.axis is not None:
                            axis = joint.axis / np.linalg.norm(joint.axis)
                            # Rodrigues' rotation formula
                            K = np.array([[0, -axis[2], axis[1]],
                                        [axis[2], 0, -axis[0]],
                                        [-axis[1], axis[0], 0]])
                            R = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * np.dot(K, K)
                            joint_transform[:3, :3] = joint_transform[:3, :3] @ R
                    
                    elif joint.joint_type == 'prismatic':
                        # Apply translation along joint axis
                        if hasattr(joint, 'axis') and joint.axis is not None:
                            # The axis is in joint's local frame, need to transform to world frame
                            local_axis = joint.axis
                            
                            # Transform the axis to world coordinates using the current joint orientation
                            world_axis = joint_transform[:3, :3] @ local_axis
                            
                            # Apply translation along the world-oriented axis
                            translation = joint_angle * world_axis
                            joint_transform[:3, 3] += translation

                # Compute joint world orientation
                joint_world_transform = parent_transform @ joint_transform
                joint_rotation_matrix = joint_world_transform[:3, :3]
                joint_euler = rotation_matrix_to_euler(joint_rotation_matrix)
                
                # Extract position from world transform
                joint_position = joint_world_transform[:3, 3]
                
                # Extract Z-axis direction (3rd column of rotation matrix)
                joint_z_direction = joint_rotation_matrix[:, 2]
                
                # Convert rotation matrix to quaternion
                joint_quaternion = rotation_matrix_to_quaternion(joint_rotation_matrix)
                joint_orientations[joint.name] = {
                    'position_xyz': joint_position,
                    'rotation_matrix': joint_rotation_matrix,
                    'euler_angles_rad': joint_euler,
                    'euler_angles_deg': np.degrees(joint_euler),
                    'joint_angle': joint_angles.get(joint.name, 0.0),
                    'joint_type': joint.joint_type,
                    'z_direction': joint_z_direction,  # Direction of Z-axis
                    'quaternion': joint_quaternion  # Quaternion (w, x, y, z)
                }
                
                # Update child link transform for next iteration
                link_transforms[child_link] = joint_world_transform
                processed_links.add(child_link)
                remaining_joints.remove(joint)
                progress_made = True
        
        if not progress_made:
            break
    
    return joint_orientations

def sort_by_number_in_name(sensor_name):
    """
    Extract the numeric part for sorting, handling both integer and string sensor IDs.
    """
    import re
    
    # Handle integer sensor IDs directly
    if isinstance(sensor_name, int):
        return (sensor_name, '')
    
    # Convert to string if needed
    sensor_name = str(sensor_name)
    
    # Find the last sequence of digits in the string
    match = re.search(r'(\d+)$', sensor_name)
    if match:
        # Extract the number and return it with the prefix
        number = int(match.group(1))
        prefix = sensor_name[:match.start()]
        return (number, prefix)
    # If no number found, return a default value
    return (float('inf'), sensor_name)

def extract_frame_data(frame):
    """
    Extract frame data handling both old and new formats
    
    Args:
        frame: Frame data (either dict with 'frameData' key or direct numpy array)
    
    Returns:
        numpy array: Processed frame data
    """
    # Handle old format (MetaSense) with frameData key
    if isinstance(frame, dict) and 'frameData' in frame:
        frame_data = frame['frameData']
    else:
        # Handle new format (nanobind) - direct numpy array
        frame_data = frame
    
    # Convert to numpy array if needed
    if not isinstance(frame_data, np.ndarray):
        frame_data = np.array(frame_data, dtype=np.float32)
    
    # Reshape to 25x25 if needed (625 elements = 25x25)
    if frame_data.size == 625:
        frame_data = frame_data.reshape(25, 25)
    elif len(frame_data.shape) == 1:
        # Try to infer square dimensions
        sqrt_size = int(np.sqrt(frame_data.size))
        if sqrt_size * sqrt_size == frame_data.size:
            frame_data = frame_data.reshape(sqrt_size, sqrt_size)
        else:
            print(f"Warning: Cannot reshape frame data of size {frame_data.size} to square")
    
    return frame_data

def load_sensor_recording(filename):
    """Load sensor recording data from a NumPy file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Recording file not found: {filename}")
    
    print(f"Loading recording from {filename}...")
    data = np.load(filename, allow_pickle=True).item()
    
    # Verify timestamps exist for all sensors
    for sensor_id, sensor_data in data.items():
        if 'timestamps' not in sensor_data or len(sensor_data['timestamps']) == 0:
            print(f"WARNING: Sensor {sensor_id} has no timestamps!")
        elif len(sensor_data['timestamps']) != len(sensor_data['frames']):
            print(f"WARNING: Sensor {sensor_id} has mismatched timestamps ({len(sensor_data['timestamps'])}) and frames ({len(sensor_data['frames'])})!")
        else:
            print(f"Sensor {sensor_id}: {len(sensor_data['frames'])} frames with timestamps")
    
    return data

def downsample_25x25_to_5x5(frame_25x25, method='min'):
    """
    Downsample a 25x25 ToF frame to 5x5 using specified pooling method
    
    The 25x25 frame is divided into 5x5 blocks as follows:
    - Block [0,0]: rows 0-4, cols 0-4
    - Block [0,1]: rows 0-4, cols 5-9  
    - Block [0,2]: rows 0-4, cols 10-14
    - Block [1,0]: rows 5-9, cols 0-4
    - etc.
    
    Args:
        frame_25x25: numpy array of shape (25, 25)
        method: pooling method - 'min', 'max', 'mean', 'median', or 'center'
    
    Returns:
        tuple: (cleaned_frame, outlier_mask, outlier_info)
            - cleaned_frame: numpy array of shape (25, 25) with outliers replaced
            - outlier_mask: boolean array showing outlier locations (True = outlier)
            - outlier_info: dict with statistics about outlier removal
    """
    if frame_25x25.shape != (25, 25):
        raise ValueError(f"Expected 25x25 frame, got {frame_25x25.shape}")
    

    frame_25x25, outlier_mask, info = filter_noise_spatial_knn(
        frame_25x25,
        k=6,                           # 6 nearest spatial neighbors
        threshold_method='iqr',        # Robust against outliers
        threshold_factor=2.0,          # Moderate sensitivity
        replacement_method='knn_median', # Stable replacement
        neighborhood='moore'           # 3x3 neighborhood
    )

    downsampled = np.zeros((5, 5))
    
    for block_row in range(5):
        for block_col in range(5):
            # Extract the 5x5 block
            row_start = block_row * 5
            row_end = row_start + 5
            col_start = block_col * 5  
            col_end = col_start + 5
            
            block = frame_25x25[row_start:row_end, col_start:col_end]
            
            # Apply the specified pooling method to this block
            if method == 'center':
                # Take center pixel (index 2,2) of the 5x5 block
                downsampled[block_row, block_col] = block[2, 2]
            elif method == 'min':
                downsampled[block_row, block_col] = np.min(block)
            elif method == 'max':
                downsampled[block_row, block_col] = np.max(block)
            elif method == 'mean':
                downsampled[block_row, block_col] = np.mean(block)
            elif method == 'median':
                downsampled[block_row, block_col] = np.median(block)
            else:
                raise ValueError(f"Unknown pooling method: {method}. Use 'min', 'max', 'mean', 'median', or 'center'")
    
    return downsampled


def filter_noise_spatial_knn(frame_25x25, k=8, threshold_method='iqr', threshold_factor=2.0, 
                            replacement_method='knn_median', neighborhood='moore'):
    """
    Filter noise from 25x25 ToF frame using spatial k-nearest neighbors
    
    This function considers the actual 2D spatial structure of the ToF sensor array,
    where neighbors are the physically adjacent pixels in the grid.
    
    Args:
        frame_25x25: numpy array of shape (25, 25) - input ToF frame
        k: int - number of nearest spatial neighbors to consider (max 24 for 5x5 window)
        threshold_method: str - method for outlier detection
            - 'iqr': Interquartile range method (recommended for ToF)
            - 'zscore': Z-score method  
            - 'mad': Median absolute deviation
            - 'std': Standard deviation from mean
        threshold_factor: float - multiplier for threshold (default: 2.0)
        replacement_method: str - how to replace outliers
            - 'knn_median': median of k nearest neighbors (recommended)
            - 'knn_mean': mean of k nearest neighbors
            - 'gaussian_blur': Gaussian-weighted neighbors
        neighborhood: str - neighbor selection method
            - 'moore': 8-connected (3x3 around pixel)
            - 'extended': 24-connected (5x5 around pixel)
            - 'cross': 4-connected (+ pattern)
    
    Returns:
        tuple: (cleaned_frame, outlier_mask, outlier_info)
            - cleaned_frame: numpy array of shape (25, 25) with outliers replaced
            - outlier_mask: boolean array showing outlier locations (True = outlier)
            - outlier_info: dict with statistics about outlier removal
    """
    
    if frame_25x25.shape != (25, 25):
        raise ValueError(f"Expected 25x25 frame, got {frame_25x25.shape}")
    
    outlier_mask = np.zeros((25, 25), dtype=bool)
    outlier_scores = np.zeros((25, 25))
    
    # Define neighborhood patterns
    if neighborhood == 'moore':
        # 3x3 neighborhood (8 neighbors)
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        max_k = 8
    elif neighborhood == 'extended':
        # 5x5 neighborhood (24 neighbors)
        offsets = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr != 0 or dc != 0:  # exclude center pixel
                    offsets.append((dr, dc))
        max_k = 24
    elif neighborhood == 'cross':
        # 4-connected neighborhood
        offsets = [(-1,0), (0,-1), (0,1), (1,0)]
        max_k = 4
    else:
        raise ValueError(f"Unknown neighborhood type: {neighborhood}")
    
    # Ensure k doesn't exceed maximum possible neighbors
    k = min(k, max_k)
    
    # Process each pixel
    for row in range(25):
        for col in range(25):
            current_value = frame_25x25[row, col]
            
            # Get valid neighbors
            neighbor_values = []
            neighbor_positions = []
            
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 25 and 0 <= nc < 25:
                    neighbor_values.append(frame_25x25[nr, nc])
                    neighbor_positions.append((nr, nc))
            
            if len(neighbor_values) < 3:  # Need minimum neighbors for statistics
                continue
            
            # Select k nearest neighbors based on value similarity
            neighbor_values = np.array(neighbor_values)
            if len(neighbor_values) > k:
                # Sort by absolute difference from current pixel
                differences = np.abs(neighbor_values - current_value)
                nearest_indices = np.argsort(differences)[:k]
                neighbor_values = neighbor_values[nearest_indices]
            
            # Calculate outlier score
            if threshold_method == 'iqr':
                if len(neighbor_values) >= 4:
                    q75, q25 = np.percentile(neighbor_values, [75, 25])
                    iqr = q75 - q25
                    median_val = np.median(neighbor_values)
                    outlier_scores[row, col] = abs(current_value - median_val) / (iqr + 1e-8)
                else:
                    outlier_scores[row, col] = 0
            
            elif threshold_method == 'zscore':
                mean_val = np.mean(neighbor_values)
                std_val = np.std(neighbor_values)
                outlier_scores[row, col] = abs(current_value - mean_val) / (std_val + 1e-8)
            
            elif threshold_method == 'mad':
                median_val = np.median(neighbor_values)
                mad = np.median(np.abs(neighbor_values - median_val))
                outlier_scores[row, col] = abs(current_value - median_val) / (mad + 1e-8)
            
            elif threshold_method == 'std':
                mean_val = np.mean(neighbor_values)
                std_val = np.std(neighbor_values)
                outlier_scores[row, col] = abs(current_value - mean_val) / (std_val + 1e-8)
    
    # Identify outliers
    outlier_mask = outlier_scores > threshold_factor
    
    # Create cleaned frame
    cleaned_frame = frame_25x25.copy()
    
    # Replace outliers
    outlier_positions = np.where(outlier_mask)
    for i in range(len(outlier_positions[0])):
        row, col = outlier_positions[0][i], outlier_positions[1][i]
        
        # Get neighbors for replacement
        neighbor_values = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 25 and 0 <= nc < 25:
                neighbor_values.append(cleaned_frame[nr, nc])
        
        if neighbor_values:
            neighbor_values = np.array(neighbor_values)
            
            # Select k nearest neighbors
            if len(neighbor_values) > k:
                current_val = frame_25x25[row, col]
                differences = np.abs(neighbor_values - current_val)
                nearest_indices = np.argsort(differences)[:k]
                neighbor_values = neighbor_values[nearest_indices]
            
            # Replace based on method
            if replacement_method == 'knn_median':
                cleaned_frame[row, col] = np.median(neighbor_values)
            elif replacement_method == 'knn_mean':
                cleaned_frame[row, col] = np.mean(neighbor_values)
            elif replacement_method == 'gaussian_blur':
                # Apply Gaussian-weighted average
                weights = np.exp(-0.5 * ((neighbor_values - np.median(neighbor_values)) / np.std(neighbor_values))**2)
                weights /= np.sum(weights)
                cleaned_frame[row, col] = np.sum(neighbor_values * weights)
    
    # Calculate statistics
    outlier_info = {
        'num_outliers': np.sum(outlier_mask),
        'outlier_percentage': (np.sum(outlier_mask) / 625) * 100,
        'outlier_scores_mean': np.mean(outlier_scores[outlier_mask]) if np.any(outlier_mask) else 0,
        'outlier_scores_max': np.max(outlier_scores),
        'threshold_used': threshold_factor,
        'method_used': threshold_method,
        'k_neighbors': k,
        'neighborhood_type': neighborhood
    }
    
    return cleaned_frame, outlier_mask, outlier_info

def load_real_tof_data(tof_file_path, frame_index=0, expected_sensors=20, pooling_method='min'):
    """
    Load real ToF sensor data and downsample to 5x5 resolution
    
    Args:
        tof_file_path: Path to the .npy ToF data file
        frame_index: Which frame to use (default: 0 for first frame)
        expected_sensors: Expected number of sensors (default: 20)
        pooling_method: Downsampling method - 'min', 'max', 'mean', or 'median' (default: 'min')
    
    Returns:
        numpy array: Shape (num_sensors, 25) - flattened 5x5 data for each sensor
    """
    print(f"Loading real ToF data from {tof_file_path}")
    print(f"Using {pooling_method} pooling for downsampling")
    
    # Load the recording data
    recording_data = load_sensor_recording(tof_file_path)
    
    # Extract active sensor IDs and sort them
    active_sensors = sorted(list(recording_data.keys()), key=sort_by_number_in_name)
    num_active_sensors = len(active_sensors)
    
    print(f"Found {num_active_sensors} active sensors in recording")
    
    if num_active_sensors != expected_sensors:
        print(f"WARNING: Expected {expected_sensors} sensors, found {num_active_sensors}")
    
    # Extract data for each sensor
    sensor_distances = []
    
    for sensor_id in active_sensors:
        sensor_data = recording_data[sensor_id]
        
        # Check if sensor has frames
        if 'frames' not in sensor_data or len(sensor_data['frames']) == 0:
            print(f"WARNING: Sensor {sensor_id} has no frames, using zeros")
            # Use zeros for missing data
            downsampled_5x5 = np.zeros((5, 5))
        else:
            # Get the specified frame (or last available frame)
            frame_idx = min(frame_index, len(sensor_data['frames']) - 1)
            frame = sensor_data['frames'][frame_idx]
            
            try:
                # Extract frame data
                frame_data = extract_frame_data(frame)
                
                # Ensure it's 25x25
                if frame_data.shape == (25, 25):
                    # Downsample from 25x25 to 5x5 using specified pooling method
                    downsampled_5x5 = downsample_25x25_to_5x5(frame_data, method=pooling_method)
                else:
                    print(f"WARNING: Sensor {sensor_id} frame shape {frame_data.shape} is not 25x25, using zeros")
                    downsampled_5x5 = np.zeros((5, 5))
                    
            except Exception as e:
                print(f"ERROR processing sensor {sensor_id}: {e}")
                downsampled_5x5 = np.zeros((5, 5))
        
        # Flatten to 1D array (25 elements) and normalize to 0-1
        flattened = downsampled_5x5.flatten()
        # Normalize pixel values (assuming 0-255 range) to 0-1
        normalized = flattened / 255.0
        sensor_distances.append(normalized)
    
    # Convert to numpy array
    distance_data = np.array(sensor_distances)  # Shape: (num_sensors, 25)
    
    print(f"Processed ToF data shape: {distance_data.shape}")
    print(f"Distance range: [{distance_data.min():.3f}, {distance_data.max():.3f}]")
    
    return distance_data

def tof_to_depth(tof):
    """Converts Time-of-Flight units back to depth (meters)"""
    tof_float = tof.astype(float) if hasattr(tof, 'astype') else float(tof)
    depth = (tof_float / 161.276162) ** 2
    return depth

def convert_ray_distance_to_position(origins, directions, quaternions, distance_pixel_normalized, resolution=[5,5]):
    """
    Convert ray distances to 3D positions
    
    Args:
        origins: List of ray origins (3D points) - shape: (num_sensors, 3)
        directions: List of ray directions (3D vectors) - shape: (num_sensors, 3)
        quaternions: List of quaternions for each sensor - shape: (num_sensors, 4)
        distance_pixel_normalized: Z-axis distance values (0-1 normalized) - shape: (num_sensors, 25)
        resolution: [width, height] grid resolution - [5, 5] = 25 rays per sensor
    
    Returns:
        List of 3D positions - shape: (num_sensors * 25, 3)
    """
    point_positions = []
    
    # Convert all distances from normalized pixel values to meters
    distances_meter = tof_to_depth(np.array(distance_pixel_normalized) * 255)  # Shape: (20, 25)
    # distances_meter = distance_pixel_normalized
    
    for sensor_idx, (origin, direction, quaternion) in enumerate(zip(origins, directions, quaternions)):
        # Normalize the direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        # Convert quaternion to rotation
        rotation = R.from_quat(quaternion)
        
        # Create local coordinate system and apply rotation
        local_forward = np.array([0.0, 0.0, 1.0])  # Z-forward
        local_right = np.array([1.0, 0.0, 0.0])    # X-right  
        local_up = np.array([0.0, 1.0, 0.0])       # Y-up
        
        # Apply rotation to get world-space basis vectors
        forward = rotation.apply(local_forward)
        right = rotation.apply(local_right)
        up = rotation.apply(local_up)
        
        # Generate FOV parameters
        h_max = np.tan(np.radians(70 / 2))  # Half horizontal FOV
        v_max = np.tan(np.radians(60 / 2))  # Half vertical FOV
        
        h_values = np.linspace(-h_max, h_max, resolution[0])  # 5 horizontal values
        v_values = np.linspace(-v_max, v_max, resolution[1])  # 5 vertical values
        
        # Generate rays for this sensor (5×5 = 25 rays)
        ray_idx = 0  # Index within this sensor's 25 rays
        
        for v in v_values:
            for h in h_values:
                # Calculate ray direction in world space
                ray_dir = forward + h * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Get the z-axis distance for this specific ray from this sensor
                z_distance = distances_meter[sensor_idx, ray_idx]
                
                # Calculate the actual distance along the ray based on z-component
                # If ray_dir·forward = cos(θ), then actual_distance = z_distance / cos(θ)
                cos_theta = np.dot(ray_dir, forward)
                
                # Avoid division by zero for rays perpendicular to forward direction
                if abs(cos_theta) < 1e-6:
                    actual_distance = float('inf')  # or handle this case as needed
                else:
                    actual_distance = z_distance / cos_theta
                
                # Calculate 3D position: origin + actual_distance * ray_direction
                position_3d = np.array(origin) + actual_distance * ray_dir
                point_positions.append(position_3d)
                
                ray_idx += 1
    
    return np.array(point_positions)  # Shape: (500, 3) for 20 sensors × 25 rays


def filter_points_by_distance(points, max_distance=5.0):
    """
    Filter points based on distance from origin
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        max_distance: maximum distance from origin to keep points
    
    Returns:
        tuple: (filtered_points, valid_indices)
    """
    distances = np.linalg.norm(points, axis=1)
    valid_mask = distances <= max_distance
    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(~valid_mask)[0]
    filtered_points = points[valid_mask]

    points[invalid_indices] = [0,0,0]
    print(f"Filtered {len(points)} points to {len(filtered_points)} points (threshold: {max_distance}m)")
    return points, valid_indices

def center_point_cloud_at_origin(points, method='centroid'):
    """
    Center point cloud at [0,0,0] using different methods
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        method: centering method - 'centroid', 'median', 'bbox_center'
    
    Returns:
        tuple: (centered_points, translation_vector)
    """
    if len(points) == 0:
        return points, np.array([0.0, 0.0, 0.0])
    
    if method == 'centroid':
        # Use mean of all points
        center = np.mean(points, axis=0)
    elif method == 'median':
        # Use median of all points (more robust to outliers)
        center = np.median(points, axis=0)
    elif method == 'bbox_center':
        # Use center of bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2.0
    else:
        raise ValueError(f"Unknown centering method: {method}")
    
    # Translate points to center at origin
    centered_points = points - center
    
    print(f"Centered point cloud using {method} method")
    print(f"Translation vector: {center}")
    print(f"Original center: {center}")
    print(f"New center: {np.mean(centered_points, axis=0)}")
    
    return centered_points, center

def filter_and_center_point_cloud(points, distance_threshold=5.0, centering_method='centroid'):
    """
    Filter points by distance threshold and center at origin
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        distance_threshold: maximum distance from origin to keep points
        centering_method: method for centering - 'centroid', 'median', 'bbox_center'
    
    Returns:
        tuple: (processed_points, filtered_indices, translation_vector)
    """
    print(f"\nProcessing point cloud with {len(points)} points...")
    
    # Step 1: Filter by distance threshold
    filtered_points, valid_indices = filter_points_by_distance(points, distance_threshold)
    
    if len(filtered_points) == 0:
        print("Warning: No points remain after filtering!")
        return np.array([]).reshape(0, 3), valid_indices, np.array([0.0, 0.0, 0.0])
    
    # Step 2: Center at origin
    # centered_points, translation_vector = center_point_cloud_at_origin(filtered_points, centering_method)
    centered_points = filtered_points
    # Print statistics
    print(f"\nPoint cloud processing complete:")
    print(f"Original points: {len(points)}")
    print(f"After distance filtering: {len(filtered_points)}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Centering method: {centering_method}")
    print(f"Final point cloud bounds:")
    if len(centered_points) > 0:
        print(f"  X: [{centered_points[:, 0].min():.3f}, {centered_points[:, 0].max():.3f}]")
        print(f"  Y: [{centered_points[:, 1].min():.3f}, {centered_points[:, 1].max():.3f}]")
        print(f"  Z: [{centered_points[:, 2].min():.3f}, {centered_points[:, 2].max():.3f}]")
    
    return centered_points, valid_indices


def remove_outliers_iqr(points, multiplier=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        multiplier: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        tuple: (filtered_points, valid_indices)
    """
    if len(points) == 0:
        return points, np.array([])
    
    # Calculate distances from origin
    distances = np.linalg.norm(points, axis=1)
    
    # Calculate IQR
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Filter points
    valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
    valid_indices = np.where(valid_mask)[0]
    filtered_points = points[valid_mask]
    
    print(f"IQR outlier removal: {len(points)} -> {len(filtered_points)} points")
    print(f"Distance bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    return filtered_points, valid_indices

def adaptive_distance_threshold(points, percentile=95):
    """
    Calculate adaptive distance threshold based on point distribution
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        percentile: percentile to use for threshold (default: 95)
    
    Returns:
        float: calculated distance threshold
    """
    if len(points) == 0:
        return 1.0
    
    distances = np.linalg.norm(points, axis=1)
    threshold = np.percentile(distances, percentile)
    
    print(f"Adaptive threshold ({percentile}th percentile): {threshold:.3f}m")
    return threshold

def process_point_cloud_complete(points, distance_threshold=None, centering_method='centroid', 
                                remove_outliers=True, outlier_multiplier=1.5):
    """
    Complete point cloud processing pipeline
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        distance_threshold: max distance threshold (None for adaptive)
        centering_method: 'centroid', 'median', or 'bbox_center'
        remove_outliers: whether to remove outliers using IQR method
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        tuple: (processed_points, processing_info)
    """
    print(f"\n{'='*60}")
    print("COMPLETE POINT CLOUD PROCESSING PIPELINE")
    print(f"{'='*60}")
    
    original_count = len(points)
    processing_info = {
        'original_count': original_count,
        'steps': []
    }
    
    if original_count == 0:
        print("No points to process!")
        return points, processing_info
    
    current_points = points.copy()
    
    # Step 1: Remove outliers (optional)
    if remove_outliers:
        print("\nStep 1: Removing outliers...")
        current_points, outlier_indices = remove_outliers_iqr(current_points, outlier_multiplier)
        processing_info['steps'].append({
            'step': 'outlier_removal',
            'points_before': len(points),
            'points_after': len(current_points),
            'valid_indices': outlier_indices
        })
    
    # Step 2: Calculate adaptive threshold if needed
    if distance_threshold is None:
        print("\nStep 2: Calculating adaptive distance threshold...")
        distance_threshold = adaptive_distance_threshold(current_points, percentile=95)
    
    # Step 3: Filter and center
    print(f"\nStep 3: Filtering and centering...")
    processed_points, filter_indices = filter_and_center_point_cloud(
        current_points, distance_threshold, centering_method
    )
    
    processing_info['steps'].append({
        'step': 'filter_and_center',
        'points_before': len(current_points),
        'points_after': len(processed_points),
        'distance_threshold': distance_threshold,
        'centering_method': centering_method,
        # 'translation_vector': translation,
        'valid_indices': filter_indices
    })
    
    # Final statistics
    final_count = len(processed_points)
    processing_info['final_count'] = final_count
    processing_info['reduction_ratio'] = (original_count - final_count) / original_count if original_count > 0 else 0
    
    print(f"\n{'='*40}")
    print("PROCESSING SUMMARY")
    print(f"{'='*40}")
    print(f"Original points: {original_count}")
    print(f"Final points: {final_count}")
    print(f"Points removed: {original_count - final_count} ({processing_info['reduction_ratio']*100:.1f}%)")
    print(f"Distance threshold used: {distance_threshold:.3f}m")
    print(f"Centering method: {centering_method}")
    
    return processed_points, processing_info



def downsample_real_tof_data(frame_data, pooling_method='min'):
    """
    Load real ToF sensor data and downsample to 5x5 resolution
    
    Args:
        tof_file_path: Path to the .npy ToF data file
        frame_index: Which frame to use (default: 0 for first frame)
        expected_sensors: Expected number of sensors (default: 20)
        pooling_method: Downsampling method - 'min', 'max', 'mean', or 'median' (default: 'min')
    
    Returns:
        numpy array: Shape (num_sensors, 25) - flattened 5x5 data for each sensor
    """

    # Load the recording data
   
    try:      
        # Ensure it's 25x25
        if frame_data.shape == (25, 25):
            # Downsample from 25x25 to 5x5 using specified pooling method
            downsampled_5x5 = downsample_25x25_to_5x5(frame_data, method=pooling_method)
        else:
            print(f"WARNING: Sensor frame shape {frame_data.shape} is not 25x25, using zeros")
            
            
    except Exception as e:
        print(f"ERROR processing sensor")
        
    # Flatten to 1D array (25 elements) and normalize to 0-1
    flattened = downsampled_5x5.flatten()
    # Normalize pixel values (assuming 0-255 range) to 0-1
    normalized = flattened / 255.0
    # Convert to numpy array
    distance_data = np.array(normalized)  # Shape: (num_sensors, 25)
    
    print(f"Processed ToF data shape: {distance_data.shape}")
    print(f"Distance range: [{distance_data.min():.3f}, {distance_data.max():.3f}]")
    
    return distance_data

def distance_to_pointcloud(robot_orientation_matrix, dof_position, real_distances, robot_position=None):
    """
    Convert distance measurements to point cloud with robot orientation transform
    
    Args:
        robot_orientation_matrix: 3x3 or 4x4 robot orientation matrix in world frame
        dof_position: List of joint angles/positions for each DOF
        real_distances: Distance measurements from ToF sensors - shape: (num_sensors, 25)
        robot_position: Robot position in world frame (optional) - shape: (3,)
    
    Returns:
        numpy array: Processed point cloud in world coordinates
    """
    # import ipdb;ipdb.set_trace()
    # Load real ToF data
    # ===================================================================
    # MAIN EXECUTION
    # ===================================================================
    real_distances=downsample_real_tof_data(real_distances,pooling_method='center')

    print(f"Robot: {robot.name}")
    print(f"Number of joints: {len(robot.joints)}")

    # Get all movable joints
    movable_joints = [joint.name for joint in robot.joints if joint.joint_type != 'fixed']
    print(f"Movable joints: {len(movable_joints)}")
    

    neutral_angles = {joint_name: dof_position[i] for i, joint_name in enumerate(movable_joints)}
    orientations_neutral = get_joint_orientations(robot, neutral_angles)

    # Display first few joints for verification
    for i, (joint_name, data) in enumerate(list(orientations_neutral.items())[:3]):
        print(f"\nJoint {i}: {joint_name}")
        print(f"Type: {data['joint_type']}")
        print(f"Position (xyz): {data['position_xyz']}")
        print(f"Z-direction: {data['z_direction']}")

    try:
        # Extract joint origins, directions, and quaternions (in robot's local frame)
        joint_origins = []
        joint_directions = []
        joint_quaternions = []

        for joint_name in movable_joints:
            if joint_name in orientations_neutral:
                data = orientations_neutral[joint_name]
                joint_origins.append(data['position_xyz'])
                joint_directions.append(data['z_direction'])
                joint_quaternions.append(data['quaternion'])  # Already in (x,y,z,w) format

        print(f"Number of joints with orientation data: {len(joint_origins)}")
        
        # Apply robot orientation transformation
        print("\n" + "="*60)
        print("APPLYING ROBOT ORIENTATION TRANSFORMATION")
        print("="*60)
        
        transformed_origins, transformed_directions, transformed_quaternions = apply_robot_orientation_transform(
            joint_origins=joint_origins,
            joint_directions=joint_directions,
            joint_quaternions=joint_quaternions,
            robot_orientation_matrix=robot_orientation_matrix,
            robot_position=robot_position
        )
        
        # Ensure we have the right number of joints for the sensor data
        num_sensors = min(len(transformed_origins), len(real_distances))
        if num_sensors < 20:
            print(f"WARNING: Only using {num_sensors} sensors (expected 20)")
        print(real_distances.shape,"==================")

        # Trim data to match available joints
        transformed_origins = transformed_origins[:num_sensors]
        transformed_directions = transformed_directions[:num_sensors]
        transformed_quaternions = transformed_quaternions[:num_sensors]
        real_distances = real_distances[:num_sensors]

        # Convert ray distances to 3D positions using transformed data
        print("\n" + "="*60)
        print("CONVERTING DISTANCES TO 3D POSITIONS")
        print("="*60)
        print(real_distances.shape,"==================")
        point_positions = convert_ray_distance_to_position(
            origins=transformed_origins,
            directions=transformed_directions, 
            quaternions=transformed_quaternions,
            distance_pixel_normalized=real_distances,
            resolution=[5, 5]
        )
        
        print(f"Generated point positions shape: {point_positions.shape}")
        
        # ===================================================================
        # 3D POINT CLOUD PROCESSING
        # ===================================================================
        
        print("\n" + "="*60)
        print("3D POINT CLOUD PROCESSING")
        print("="*60)
        
        processed_points, processing_info = process_point_cloud_complete(
            point_positions,
            distance_threshold=1.5,
            centering_method='centroid',
            remove_outliers=True,
            outlier_multiplier=1.5
        )
        
        return processed_points
        
    except Exception as e:
        print(f"Error processing ToF data: {e}")
        print("Returning empty point cloud...")
        return np.array([]).reshape(0, 3)