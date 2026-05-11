import os
import torch
import numpy as np
from torch.utils.data import Dataset

class NTUSkeletonDataset(Dataset):
    def __init__(self, data_dir, seq_len=60, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        
        # Parse files
        if os.path.exists(data_dir):
            for file_name in os.listdir(data_dir):
                if file_name.endswith('.skeleton'):
                    # Extract action class from filename (Axxx)
                    # Example: S001C001P001R001A043.skeleton
                    action_str = file_name.split('A')[1][:3]
                    action_id = int(action_str)
                    
                    # NTU action 43 is falling down
                    label = 1 if action_id == 43 else 0
                    self.samples.append((os.path.join(data_dir, file_name), label))

    def __len__(self):
        return len(self.samples)
        
    def read_skeleton_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            return np.zeros((self.seq_len, 25, 3))
            
        frame_count = int(lines[0].strip())
        current_line = 1
        
        frames = []
        
        for _ in range(frame_count):
            if current_line >= len(lines):
                break
                
            body_count = int(lines[current_line].strip())
            current_line += 1
            
            if body_count == 0:
                frames.append(np.zeros((25, 3)))
                continue
                
            # Just take the first body
            body_info = lines[current_line].strip().split()
            current_line += 1
            
            joint_count = int(lines[current_line].strip())
            current_line += 1
            
            joints = []
            for _ in range(joint_count):
                joint_info = lines[current_line].strip().split()
                x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                joints.append([x, y, z])
                current_line += 1
                
            frames.append(np.array(joints))
            
            # Skip any remaining bodies in this frame
            for _ in range(1, body_count):
                body_info = lines[current_line].strip().split()
                current_line += 1
                joint_count = int(lines[current_line].strip())
                current_line += 1 + joint_count

        frames = np.array(frames) # (T, 25, 3)
        if len(frames) == 0:
            return np.zeros((self.seq_len, 25, 3))
            
        return frames

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Shape: (T, 25, 3)
        pose_seq = self.read_skeleton_file(file_path)
        
        # Compute velocity: V_t = P_t - P_{t-1}
        if len(pose_seq) > 1:
            velocity = pose_seq[1:] - pose_seq[:-1] # (T-1, 25, 3)
        else:
            velocity = np.zeros((1, 25, 3))
            
        # Pad or truncate to seq_len
        T = velocity.shape[0]
        if T > self.seq_len:
            # Truncate
            start = (T - self.seq_len) // 2
            velocity = velocity[start:start + self.seq_len]
        elif T < self.seq_len:
            # Zero pad
            pad_len = self.seq_len - T
            pad_arr = np.zeros((pad_len, 25, 3))
            velocity = np.vstack((velocity, pad_arr))
            
        # Flatten joint and dim dimensions: (seq_len, 75)
        velocity = velocity.reshape(self.seq_len, -1)
        velocity = torch.tensor(velocity, dtype=torch.float32)
        
        if self.transform:
            velocity = self.transform(velocity)
            
        return velocity, label
