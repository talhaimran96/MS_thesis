import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.kinetics_dataset import KineticsDataset
from src.models.videomae import VideoMAE
from src.utils.logger import ExperimentLogger
from Experiments.train_stream_a import compute_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    branch_name = "stream_a_rgb_baseline"
    model_type = "videomae_v2"
    
    # We load the hyperparams to recreate the exact model
    results_dir = os.path.join('Results', f"{branch_name}_{model_type}")
    hp_path = os.path.join(results_dir, 'hyperparameters.json')
    
    if not os.path.exists(hp_path):
        print(f"Error: Could not find hyperparameters at {hp_path}. Run training first.")
        return
        
    import json
    with open(hp_path, 'r') as f:
        hyperparams = json.load(f)
        
    print("Loading Validation Dataset...")
    data_dir = 'Data/raw/kinetics400_5per/train/'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    full_dataset = KineticsDataset(data_dir=data_dir, num_frames=16, frame_stride=4, transform=transform)
    
    eval_loader = DataLoader(full_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4, drop_last=True)
    
    print("Initializing Model...")
    model = VideoMAE(depth=hyperparams['depth'], num_heads=hyperparams['num_heads'])
    
    # Load Best Model Weights
    models_dir = os.path.join('Data', 'models', branch_name)
    best_model_path = os.path.join(models_dir, f"{model_type}_best.pth")
    
    if not os.path.exists(best_model_path):
        print(f"Error: Could not find best model weights at {best_model_path}.")
        return
        
    print(f"Loading weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print("Starting Evaluation...")
    total_loss = 0.0
    pbar = tqdm(eval_loader, desc="Evaluation")
    with torch.no_grad():
        for frames, labels in pbar:
            frames = frames.permute(0, 2, 1, 3, 4).to(device)
            pred, mask = model(frames, mask_ratio=hyperparams['mask_ratio'])
            loss = compute_loss(pred, frames, mask)
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
    avg_loss = total_loss / len(eval_loader)
    print(f"Evaluation Complete. Average Reconstruction Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()
