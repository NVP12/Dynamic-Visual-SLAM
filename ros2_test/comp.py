import numpy as np

def read_trajectory(file_path):
    """Read trajectory data from a txt file."""
    trajectory = []
    with open(file_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            timestamp = data[0]
            position = np.array(data[1:4])
            orientation = np.array(data[4:8])  # Assuming quaternion
            trajectory.append((timestamp, position, orientation))
    return trajectory

def compute_translation_error(trajectory1, trajectory2):
    """Compute the RMSE of the translation errors."""
    assert len(trajectory1) == len(trajectory2)
    errors = []
    for (timestamp1, pos1, _), (timestamp2, pos2, _) in zip(trajectory1, trajectory2):
        if timestamp1 == timestamp2:
            error = np.linalg.norm(pos1 - pos2)
            errors.append(error)
    return np.sqrt(np.mean(np.square(errors)))

def compute_rotation_error(trajectory1, trajectory2):
    """Compute the average rotation error (in radians) between two trajectories."""
    assert len(trajectory1) == len(trajectory2)
    errors = []
    for (_, _, quat1), (_, _, quat2) in zip(trajectory1, trajectory2):
        # Quaternion dot product to find the angle between quaternions
        dot_product = np.dot(quat1, quat2)
        # Ensure the dot product is between -1 and 1 due to floating point precision
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = 2 * np.arccos(dot_product)
        errors.append(angle)
    return np.mean(errors)

# Load your SLAM trajectory and TUM ground truth
slam_trajectory = read_trajectory("MonocularTrajectory.txt")
ground_truth_trajectory = read_trajectory("groundtruth.txt")

# Compute the errors
translation_error = compute_translation_error(slam_trajectory, ground_truth_trajectory)
rotation_error = compute_rotation_error(slam_trajectory, ground_truth_trajectory)

print(f"Translation Error (RMSE): {translation_error:.4f} meters")
print(f"Rotation Error (Average): {rotation_error:.4f} radians")

