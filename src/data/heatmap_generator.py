import numpy as np

def generate_3d_heatmap(skeleton_sequence, target_shape=(32, 32, 32), sigma=2.0):
    """
    Generates a dense 3D Spatial-Temporal Heatmap Volume from a sequence of 3D joints.
    
    Args:
        skeleton_sequence: np.ndarray of shape (T, V, 3) where T is frames, V is joints.
                           (Assumes a single body).
        target_shape: Tuple (D, H, W) for the 3D volume spatial dimensions.
        sigma: Variance of the 3D Gaussian.
        
    Returns:
        np.ndarray of shape (T, D, H, W) representing the heatmap volume.
    """
    T, V, _ = skeleton_sequence.shape
    D, H, W = target_shape
    
    heatmaps = np.zeros((T, D, H, W), dtype=np.float32)
    
    # Normalize coordinates to fit within the [0, D-1], [0, H-1], [0, W-1] bounding box.
    # Standardize data to [-1, 1] first (assuming it's roughly centered and scaled)
    # This is a simplistic normalization. In practice, min-max across the dataset is better.
    min_vals = np.min(skeleton_sequence, axis=(0, 1), keepdims=True)
    max_vals = np.max(skeleton_sequence, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    norm_seq = (skeleton_sequence - min_vals) / range_vals # [0, 1]
    
    # Scale to volume dimensions
    scaled_seq = np.zeros_like(norm_seq)
    scaled_seq[..., 0] = norm_seq[..., 0] * (W - 1) # X maps to W
    scaled_seq[..., 1] = norm_seq[..., 1] * (H - 1) # Y maps to H
    scaled_seq[..., 2] = norm_seq[..., 2] * (D - 1) # Z maps to D
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    
    for t in range(T):
        for v in range(V):
            x, y, z = scaled_seq[t, v]
            
            # Skip invalid/empty joints (e.g. if all coords are 0 due to padding)
            if np.all(skeleton_sequence[t, v] == 0):
                continue
                
            # Accumulate Gaussians
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2 + (zz - z)**2) / (2 * sigma**2))
            heatmaps[t] = np.maximum(heatmaps[t], gaussian)
            
    return heatmaps
