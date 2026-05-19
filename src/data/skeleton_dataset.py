import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from .ntu_parser import read_skeleton_file, center_skeleton
from .heatmap_generator import generate_3d_heatmap

class NTUSkeletonDataset(Dataset):
    """
    Dataset for NTU RGB+D skeletons.
    Provides either raw graph sequences (for ST-GCN) or 3D Heatmaps (for STH-MAE).
    """
    def __init__(self, data_dir: str, mode: str = 'heatmap', max_frames: int = 100, target_shape=(32, 32, 32)):
        """
        Args:
            data_dir: Directory containing .skeleton files.
            mode: 'heatmap' for STH-MAE, 'graph' for ST-GCN.
            max_frames: Max frames to sample/pad to.
            target_shape: Spatial dimensions for the 3D heatmap (D, H, W).
        """
        self.data_dir = data_dir
        self.mode = mode
        self.max_frames = max_frames
        self.target_shape = target_shape
        
        # Load file list
        self.file_paths = glob.glob(os.path.join(data_dir, "**", "*.skeleton"), recursive=True)
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Parse raw data (F, M, V, 3)
        raw_data = read_skeleton_file(file_path, max_bodies=1, num_joints=25)
        
        # Center the skeleton
        data = center_skeleton(raw_data)
        
        # Squeeze the body dimension since we assumed max_bodies=1 for simple pipeline
        # (F, V, 3)
        if data.shape[0] > 0:
            data = data[:, 0, :, :]
        else:
            data = np.zeros((1, 25, 3))
            
        # Temporal sampling / padding
        F = data.shape[0]
        if F > self.max_frames:
            # Uniformly sample frames
            indices = np.linspace(0, F - 1, self.max_frames, dtype=int)
            data = data[indices]
        elif F < self.max_frames:
            # Pad with zeros
            pad_amount = self.max_frames - F
            padding = np.zeros((pad_amount, 25, 3))
            data = np.concatenate([data, padding], axis=0)
            
        if self.mode == 'heatmap':
            heatmap = generate_3d_heatmap(data, target_shape=self.target_shape)
            # Shape: (T, D, H, W) -> add channel dim for 3D CNN (C, T, D, H, W) where C=1
            heatmap = np.expand_dims(heatmap, axis=0)
            return torch.tensor(heatmap, dtype=torch.float32)
            
        elif self.mode == 'graph':
            # Shape: (T, V, 3) -> reshape for GCN: usually (C, T, V, M)
            # C=3, T=frames, V=joints, M=bodies=1
            graph_data = np.transpose(data, (2, 0, 1)) # (3, T, V)
            graph_data = np.expand_dims(graph_data, axis=-1) # (3, T, V, 1)
            return torch.tensor(graph_data, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
