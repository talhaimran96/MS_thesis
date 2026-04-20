import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

class KineticsDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_stride=4, transform=None):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for vid_name in os.listdir(cls_dir):
                if vid_name.endswith('.mp4') or vid_name.endswith('.avi'):
                    self.samples.append((os.path.join(cls_dir, vid_name), self.class_to_idx[cls_name]))
                    
    def __len__(self):
        return len(self.samples)

    def _get_frame_indices(self, total_frames):
        # We need num_frames frames with frame_stride
        segment_len = self.num_frames * self.frame_stride
        if total_frames > segment_len:
            start = random.randint(0, total_frames - segment_len)
            indices = list(range(start, start + segment_len, self.frame_stride))
        elif total_frames > self.num_frames:
            # If video is shorter than segment_len but has at least num_frames
            # Uniformly sample num_frames
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
        else:
            # If video has fewer than num_frames, pad by repeating the last frame
            indices = list(range(total_frames))
            indices += [indices[-1]] * (self.num_frames - total_frames)
        return indices
        
    def __getitem__(self, idx):
        while True:
            try:
                vid_path, label = self.samples[idx]
                vr = VideoReader(vid_path, ctx=cpu(0))
                total_frames = len(vr)
                
                indices = self._get_frame_indices(total_frames)
                frames = vr.get_batch(indices) # Shape: (T, H, W, C)
                
                # Convert to float and [0, 1] range
                frames = frames.float() / 255.0
                
                # Transform expects (C, T, H, W) or we apply transform per frame
                # Let's reshape to (T, C, H, W) for torchvision transforms
                frames = frames.permute(0, 3, 1, 2)
                
                if self.transform:
                    frames = self.transform(frames)
                    
                return frames, label
            except Exception as e:
                # If decord fails to read the video, randomly sample another video
                idx = random.randint(0, len(self.samples) - 1)
