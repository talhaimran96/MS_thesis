import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import mediapipe as mp

class VelocityMediaPipeDataset(Dataset):
    def __init__(self, data_dir, seq_len=60, transform=None, binary_mode=True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.binary_mode = binary_mode
        self.samples = []
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        pseudo_falls = {'faceplanting', 'diving cliff', 'bungee jumping', 'springboard diving', 'somersaulting', 'parkour'}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if self.binary_mode:
                label = 1 if cls_name in pseudo_falls else 0
            else:
                label = self.class_to_idx[cls_name]
                
            for vid_name in os.listdir(cls_dir):
                if vid_name.endswith('.mp4') or vid_name.endswith('.avi'):
                    self.samples.append((os.path.join(cls_dir, vid_name), label))

        self.model_asset_path = 'pose_landmarker_lite.task'
        self.cache_dir = os.path.join(data_dir, '..', 'mediapipe_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.samples)
        
    def get_pose_sequence(self, vid_path):
        vid_basename = os.path.basename(vid_path)
        cache_path = os.path.join(self.cache_dir, vid_basename + '.npy')
        
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except:
                pass
                
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return np.zeros((1, 33, 3))
            
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
            
        try:
            detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            return np.zeros((1, 33, 3))
        
        poses = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            try:
                detection_result = detector.detect(mp_image)
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = []
                    for lm in detection_result.pose_landmarks[0]:
                        landmarks.append([lm.x, lm.y, lm.z])
                    poses.append(landmarks)
                else:
                    poses.append(np.zeros((33, 3)))
            except Exception:
                poses.append(np.zeros((33, 3)))
                
        cap.release()
        
        if len(poses) == 0:
            result = np.zeros((1, 33, 3))
        else:
            result = np.array(poses) # (T, 33, 3)
            
        # Save to cache
        np.save(cache_path, result)
        return result

    def __getitem__(self, idx):
        # MediaPipe might randomly fail on corrupted videos, try multiple times
        max_tries = 5
        for _ in range(max_tries):
            try:
                vid_path, label = self.samples[idx]
                
                # Shape: (T, 33, 3)
                pose_seq = self.get_pose_sequence(vid_path)
                
                # Compute velocity: V_t = P_t - P_{t-1}
                if len(pose_seq) > 1:
                    velocity = pose_seq[1:] - pose_seq[:-1] # (T-1, 33, 3)
                else:
                    velocity = np.zeros((1, 33, 3))
                    
                # Pad or truncate to seq_len
                T = velocity.shape[0]
                if T > self.seq_len:
                    start = (T - self.seq_len) // 2
                    velocity = velocity[start:start + self.seq_len]
                elif T < self.seq_len:
                    pad_len = self.seq_len - T
                    pad_arr = np.zeros((pad_len, 33, 3))
                    velocity = np.vstack((velocity, pad_arr))
                    
                # Flatten: (seq_len, 99)
                velocity = velocity.reshape(self.seq_len, -1)
                velocity = torch.tensor(velocity, dtype=torch.float32)
                
                if self.transform:
                    velocity = self.transform(velocity)
                    
                return velocity, label
                
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)
                
        return torch.zeros((self.seq_len, 99), dtype=torch.float32), 0
