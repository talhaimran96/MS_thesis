import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.data.kinetics_dataset import KineticsDataset
from src.models.videomae import VideoMAE
from src.utils.logger import ExperimentLogger

def compute_loss(pred, target, mask, patch_size=16, tube_size=2):
    B, C, T, H, W = target.shape
    p = patch_size
    t = tube_size
    h_p = H // p
    w_p = W // p
    t_p = T // t
    
    target = target.reshape(B, C, t_p, t, h_p, p, w_p, p)
    target = target.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, t_p*h_p*w_p, t*p*p*C)
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def main():
    parser = argparse.ArgumentParser(description="Train Stream A VideoMAE")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g., Data/models/stream_a_rgb_baseline/videomae_v2_last.pth)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    hyperparams = {
        'batch_size': 4,
        'epochs': 50,
        'lr': 1e-4,
        'mask_ratio': 0.9,
        'depth': 4,
        'num_heads': 6,
        'patch_size': 16,
        'tube_size': 2
    }
    
    logger = ExperimentLogger(branch_name="stream_a_rgb_baseline", 
                              model_type="videomae_v2", 
                              hyperparams=hyperparams,
                              resume=(args.resume is not None))
    
    # Data
    data_dir = 'Data/raw/kinetics400_5per/train/'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    print("Loading Dataset...")
    full_dataset = KineticsDataset(data_dir=data_dir, num_frames=16, frame_stride=4, transform=transform)
    
    # Split into 80/20 train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4, drop_last=True)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    print("Initializing Model...")
    model = VideoMAE(depth=hyperparams['depth'], num_heads=hyperparams['num_heads'])
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'])
    
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = len(logger.val_losses)
            best_val_loss = min(logger.val_losses) if logger.val_losses else float('inf')
            print(f"Loaded old-format model weights from '{args.resume}'. Inferred start_epoch={start_epoch}")
    
    print("Starting Training...")
    for epoch in range(start_epoch, hyperparams['epochs']):
        # Train Loop
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Train]")
        for batch_idx, (frames, labels) in enumerate(pbar_train):
            frames = frames.permute(0, 2, 1, 3, 4).to(device)
            
            optimizer.zero_grad()
            pred, mask = model(frames, mask_ratio=hyperparams['mask_ratio'])
            loss = compute_loss(pred, frames, mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # For quick testing, we can break early
            # if batch_idx > 10:
            #     break
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Val]")
        with torch.no_grad():
            for batch_idx, (frames, labels) in enumerate(pbar_val):
                frames = frames.permute(0, 2, 1, 3, 4).to(device)
                pred, mask = model(frames, mask_ratio=hyperparams['mask_ratio'])
                loss = compute_loss(pred, frames, mask)
                val_loss += loss.item()
                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}"})
                
                # For quick testing, we can break early
                # if batch_idx > 10:
                #     break
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Log to file and plot
        logger.log_epoch(epoch + 1, avg_train_loss, avg_val_loss)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, logger.get_best_model_path())
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
            
    # Save Last Model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, logger.get_last_model_path())
    print("Training loop complete. Models and logs saved.")

if __name__ == '__main__':
    main()
