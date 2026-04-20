import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from src.data.kinetics_dataset import KineticsDataset
from src.models.videomae import VideoMAE

def main():
    data_dir = 'Data/raw/kinetics400_5per/train/'
    
    # Simple transform: Resize and CenterCrop
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    print("Initializing KineticsDataset...")
    dataset = KineticsDataset(data_dir=data_dir, num_frames=16, frame_stride=4, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    if len(dataset) > 0:
        frames, label = dataset[0]
        # frames shape is (T, C, H, W)
        print(f"Extracted frames shape: {frames.shape}")
        
        # Prepare for VideoMAE: (B, C, T, H, W)
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
        print(f"Input shape for VideoMAE: {frames.shape}")
        
        print("Initializing VideoMAE skeleton...")
        model = VideoMAE()
        
        print("Forward pass...")
        out = model(frames)
        print(f"Output shape: {out.shape}")
        print("Pipeline test successful!")
    else:
        print("No videos found to test.")

if __name__ == '__main__':
    main()
