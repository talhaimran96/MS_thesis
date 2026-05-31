import os
import random
import glob
import torch
import json
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

class GMDCSA24VideoDataset(Dataset):
    def __init__(self, root_dir, test_subject=1, split='train', split_ratio=1.0, num_frames=16, frame_size=224):
        """
        GMDCSA-24 Fall Detection Dataset loader supporting K-Fold CV (Leave-One-Subject-Out).
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        
        self.video_paths = []
        self.labels = []
        
        subjects = [f"Subject {i}" for i in range(1, 5)]
        
        if split == 'train':
            # Use all subjects except the test_subject
            target_subjects = [s for i, s in enumerate(subjects) if (i+1) != test_subject]
        else:
            # Use only the test_subject
            target_subjects = [f"Subject {test_subject}"]
            
        for subj in target_subjects:
            subj_dir = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_dir): continue
            
            # ADL is class 0, Fall is class 1
            for cls_name, cls_idx in [("ADL", 0), ("Fall", 1)]:
                cls_dir = os.path.join(subj_dir, cls_name)
                if not os.path.isdir(cls_dir): continue
                videos = sorted(glob.glob(os.path.join(cls_dir, '*.mp4')))
                
                # Apply data ablation split if train
                if split == 'train' and split_ratio < 1.0:
                    num_keep = max(1, int(len(videos) * split_ratio))
                    # Seed random to ensure the same split is selected for ablation consistency
                    random.Random(42).shuffle(videos)
                    videos = videos[:num_keep]
                    
                for v in videos:
                    self.video_paths.append(v)
                    self.labels.append(cls_idx)
                    
    def __len__(self):
        return len(self.video_paths)
        
    def _read_video_av(self, video_path):
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
        if len(frames) == 0:
            raise ValueError("No frames found")
        return torch.stack(frames)
        
    def transform_video(self, video_tensor):
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        transformed_frames = []
        for i in range(video_tensor.size(0)):
            frame = video_tensor[i].float() / 255.0
            frame = F.resize(frame, [self.frame_size, self.frame_size], antialias=True)
            frame = F.normalize(frame, mean=mean, std=std)
            transformed_frames.append(frame)
        return torch.stack(transformed_frames)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            vframes = self._read_video_av(video_path)
            total_frames = vframes.size(0)
            
            if total_frames < self.num_frames:
                pad_size = self.num_frames - total_frames
                pad_frames = vframes[-1:].repeat(pad_size, 1, 1, 1)
                vframes = torch.cat([vframes, pad_frames], dim=0)
                start_idx = 0
            else:
                if self.split == 'train':
                    start_idx = random.randint(0, total_frames - self.num_frames)
                else:
                    start_idx = (total_frames - self.num_frames) // 2
                    
            sampled_frames = vframes[start_idx:start_idx + self.num_frames]
            video_tensor = self.transform_video(sampled_frames)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size), label

class OOPSVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, frame_size=224):
        """
        OOPS Dataset loader for quarantined testing.
        Uses the transition_times.json to derive binary labels:
        - Class 0 (ADL): n_notfound >= 2
        - Class 1 (Fall/Failure): n_notfound <= 1
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        self.video_paths = []
        self.labels = []
        
        val_txt_path = os.path.join(root_dir, 'annotations', 'val.txt')
        json_path = os.path.join(root_dir, 'annotations', 'transition_times.json')
        video_dir = os.path.join(root_dir, 'video', 'oops_video', 'val')
        
        if not os.path.exists(val_txt_path) or not os.path.exists(json_path) or not os.path.exists(video_dir):
            print("WARNING: OOPS dataset missing. Please check Data/raw/oops_dataset/")
            return
            
        with open(json_path, 'r') as f:
            transition_data = json.load(f)
            
        with open(val_txt_path, 'r') as f:
            val_videos = [line.strip() for line in f.readlines()]
            
        for vid_id in val_videos:
            if vid_id not in transition_data: continue
            
            n_notfound = transition_data[vid_id].get('n_notfound', 0)
            if n_notfound >= 2:
                label = 0 # ADL
            else:
                label = 1 # Fall/Failure
                
            vid_path = os.path.join(video_dir, f"{vid_id}.mp4")
            if os.path.exists(vid_path):
                self.video_paths.append(vid_path)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.video_paths)
        
    def _read_video_av(self, video_path):
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
        if len(frames) == 0:
            raise ValueError("No frames found")
        return torch.stack(frames)
        
    def transform_video(self, video_tensor):
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        transformed_frames = []
        for i in range(video_tensor.size(0)):
            frame = video_tensor[i].float() / 255.0
            frame = F.resize(frame, [self.frame_size, self.frame_size], antialias=True)
            frame = F.normalize(frame, mean=mean, std=std)
            transformed_frames.append(frame)
        return torch.stack(transformed_frames)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            vframes = self._read_video_av(video_path)
            total_frames = vframes.size(0)
            
            if total_frames < self.num_frames:
                pad_size = self.num_frames - total_frames
                pad_frames = vframes[-1:].repeat(pad_size, 1, 1, 1)
                vframes = torch.cat([vframes, pad_frames], dim=0)
                start_idx = 0
            else:
                # Center crop for evaluation
                start_idx = (total_frames - self.num_frames) // 2
                    
            sampled_frames = vframes[start_idx:start_idx + self.num_frames]
            video_tensor = self.transform_video(sampled_frames)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size), label
