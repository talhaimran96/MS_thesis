import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from .ntu_parser import center_skeleton
from .heatmap_generator import generate_3d_heatmap
from .video_processor import MediaPipePoseExtractor

class GMDCSASkeletonDataset(Dataset):
    """
    Dataset for GMDCSA24 video dataset.
    Extracts 3D skeleton frames using MediaPipe and formats them as either graphs or 3D Heatmaps.
    Processed skeleton data is kept in memory to avoid repeated extraction.
    """
    def __init__(self, data_dir: str, mode: str = 'heatmap', max_frames: int = 100, target_shape=(32, 32, 32)):
        self.data_dir = data_dir
        self.mode = mode
        self.max_frames = max_frames
        self.target_shape = target_shape
        
        self.extractor = MediaPipePoseExtractor()
        
        # In-memory cache for processed skeleton numpy arrays
        self.memory_cache = {}
        
        # Load file list and extract labels from parent directory name
        # Fall = 1, ADL = 0
        self.samples = []
        
        # Recursively search for mp4 files
        video_paths = glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True)
        
        for v_path in video_paths:
            # Determine label from directory name
            parent_dir = os.path.basename(os.path.dirname(v_path))
            if parent_dir.lower() == 'fall':
                label = 1
            elif parent_dir.lower() == 'adl':
                label = 0
            else:
                continue # Skip if it doesn't belong to a known class
                
            self.samples.append((v_path, label))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        v_path, label = self.samples[idx]
        
        # Retrieve from cache or process on the fly
        if v_path in self.memory_cache:
            data = self.memory_cache[v_path]
        else:
            # data is (F, V, 3) because extract_from_video returns (num_frames, 25, 3)
            data = self.extractor.extract_from_video(v_path)
            
            # To use center_skeleton (which expects F, M, V, 3), we reshape briefly
            data_expanded = np.expand_dims(data, axis=1) # (F, 1, 25, 3)
            data_expanded = center_skeleton(data_expanded)
            data = data_expanded[:, 0, :, :]
            
            # Store to cache
            self.memory_cache[v_path] = data
            
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
            return torch.tensor(heatmap, dtype=torch.float32), label
            
        elif self.mode == 'graph':
            # Shape: (T, V, 3) -> reshape for GCN: usually (C, T, V, M)
            # C=3, T=frames, V=joints, M=bodies=1
            graph_data = np.transpose(data, (2, 0, 1)) # (3, T, V)
            graph_data = np.expand_dims(graph_data, axis=-1) # (3, T, V, 1)
            return torch.tensor(graph_data, dtype=torch.float32), label
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
