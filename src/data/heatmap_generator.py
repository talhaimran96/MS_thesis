import numpy as np

def generate_3d_heatmap(skeleton_sequence, target_shape=(32, 32, 32), sigma=2.0):
    """
    Generates a 3D Spatial-Temporal Heatmap Volume (T, H, W) from a sequence of 3D joints.
    We project the 3D joints (X,Y,Z) to 2D (X,Y) and stack them over time T.
    
    Args:
        skeleton_sequence: np.ndarray of shape (T, V, 3)
        target_shape: Tuple (T_target, H, W)
        sigma: Variance of the Gaussian.
        
    Returns:
        np.ndarray of shape (T_target, H, W)
    """
    T_target, H, W = target_shape
    T, V, _ = skeleton_sequence.shape
    
    heatmaps = np.zeros((T_target, H, W), dtype=np.float32)
    
    min_vals = np.min(skeleton_sequence, axis=(0, 1), keepdims=True)
    max_vals = np.max(skeleton_sequence, axis=(0, 1), keepdims=True)
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    norm_seq = (skeleton_sequence - min_vals) / range_vals 
    
    scaled_seq = np.zeros_like(norm_seq)
    scaled_seq[..., 0] = norm_seq[..., 0] * (W - 1) 
    scaled_seq[..., 1] = norm_seq[..., 1] * (H - 1) 
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for t in range(min(T, T_target)):
        for v in range(V):
            x, y, z = scaled_seq[t, v]
            if np.all(skeleton_sequence[t, v] == 0):
                continue
                
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            heatmaps[t] = np.maximum(heatmaps[t], gaussian)
            
    return heatmaps
