import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from tqdm import tqdm
from src.data.velocity_dataset import VelocityMediaPipeDataset

def main():
    parser = argparse.ArgumentParser(description="Prepare MediaPipe Velocity Dataset")
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length for velocity vectors')
    args = parser.parse_args()

    data_dir = 'Data/raw/kinetics400_5per/train/'
    print(f"Loading MediaPipe Video Dataset from {data_dir}...")
    full_dataset = VelocityMediaPipeDataset(data_dir=data_dir, seq_len=args.seq_len, binary_mode=True)
    
    print(f"Total videos to process: {len(full_dataset.samples)}")
    print(f"Cache directory: {full_dataset.cache_dir}")
    
    # Process sequentially with tqdm
    for idx in tqdm(range(len(full_dataset.samples)), desc="Extracting MediaPipe Features"):
        vid_path, _ = full_dataset.samples[idx]
        _ = full_dataset.get_pose_sequence(vid_path)

    print("MediaPipe feature extraction and caching complete.")

if __name__ == '__main__':
    main()
