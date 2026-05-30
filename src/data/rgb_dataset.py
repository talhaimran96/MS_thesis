import os
import random
import glob
import torch
from torch.utils.data import Dataset
import av
import torchvision.transforms as T
from torchvision.transforms import functional as F

class KineticsVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, frame_size=224, split='train'):
        """
        Args:
            root_dir (str): Path to Kinetics dataset (e.g., Data/raw/kinetics400_5per/train).
            num_frames (int): Number of contiguous frames to sample per video.
            frame_size (int): Spatial size (H, W) to crop/resize frames.
            split (str): 'train' or 'val'.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        
        # Load video paths
        self.video_paths = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            videos = glob.glob(os.path.join(cls_dir, '*.mp4'))
            for v in videos:
                self.video_paths.append(v)
                self.labels.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.video_paths)
    
    def transform_video(self, video_tensor):
        # video_tensor shape: (T, H, W, C)
        
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # Normalization values commonly used for Kinetics
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        
        transformed_frames = []
        for i in range(video_tensor.size(0)):
            frame = video_tensor[i]
            
            # Convert to float and scale to [0, 1]
            frame = frame.float() / 255.0
            
            # Resize
            frame = F.resize(frame, [self.frame_size, self.frame_size], antialias=True)
            
            # Normalize
            frame = F.normalize(frame, mean=mean, std=std)
            
            transformed_frames.append(frame)
            
        return torch.stack(transformed_frames) # (T, C, H, W)

    def _read_video_av(self, video_path):
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
        if len(frames) == 0:
            raise ValueError("No frames found")
        return torch.stack(frames) # (T, H, W, C)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Read video using PyAV
            vframes = self._read_video_av(video_path)
            
            total_frames = vframes.size(0)
            
            if total_frames == 0:
                raise ValueError("Video has 0 frames")
                
            if total_frames < self.num_frames:
                # Pad by repeating the last frame
                pad_size = self.num_frames - total_frames
                pad_frames = vframes[-1:].repeat(pad_size, 1, 1, 1)
                vframes = torch.cat([vframes, pad_frames], dim=0)
                start_idx = 0
            else:
                if self.split == 'train':
                    # Random temporal crop
                    start_idx = random.randint(0, total_frames - self.num_frames)
                else:
                    # Center temporal crop
                    start_idx = (total_frames - self.num_frames) // 2
                    
            sampled_frames = vframes[start_idx:start_idx + self.num_frames]
            
            # Transform
            video_tensor = self.transform_video(sampled_frames)
            
            # Transpose to (C, T, H, W) for standard 3D CNN / ViT inputs
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            
            return video_tensor, label
            
        except Exception as e:
            # Fallback for corrupted videos
            print(f"Error loading {video_path}: {e}")
            # Return a zero tensor and the label
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size), label
