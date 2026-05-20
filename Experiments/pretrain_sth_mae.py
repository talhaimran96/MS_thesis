import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import sys

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sth_mae import STH_MAE
from src.data.skeleton_dataset import NTUSkeletonDataset
from src.utils.logger import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description="Pretrain STH-MAE")
    parser.add_argument('--data_dir', type=str, default='Data/raw/nturgbd_skeletons_s001_to_s017')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    logger = ExperimentLogger(experiment_name="sth_mae_pretrain")
    logger.log_hyperparams(vars(args))
    
    dataset = NTUSkeletonDataset(args.data_dir, mode='heatmap', max_frames=32, target_shape=(32, 32, 32))
    
    if len(dataset) == 0:
        logger.log_info("WARNING: Dataset is empty. Check data_dir.")
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    model = STH_MAE(target_shape=(32, 32, 32), patch_size=(4, 4, 4), in_channels=1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.log_info(f"Resumed from {args.resume} at epoch {start_epoch}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.log_info("Starting training loop...")
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss, _, _ = model(batch, mask_ratio=args.mask_ratio)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(len(dataloader), 1)
        logger.log_epoch(epoch, avg_loss)
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        logger.save_model(model, optimizer, epoch, is_best=is_best)
        
    logger.plot_losses()
    logger.log_info("Training complete.")

if __name__ == "__main__":
    main()
