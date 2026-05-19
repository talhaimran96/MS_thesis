import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.st_gcn import ST_GCN
from src.data.skeleton_dataset import NTUSkeletonDataset
from src.utils.logger import ExperimentLogger

def nt_xent_loss(out_1, out_2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss for contrastive learning."""
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)
    
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def main():
    parser = argparse.ArgumentParser(description="Pretrain ST-GCN via Contrastive Learning")
    parser.add_argument('--data_dir', type=str, default='Data/raw/nturgbd_skeletons_s001_to_s017')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    logger = ExperimentLogger(experiment_name="st_gcn_pretrain")
    logger.log_hyperparams(vars(args))
    
    dataset = NTUSkeletonDataset(args.data_dir, mode='graph')
    # If dataset is empty for testing, handle gracefully or use a dummy size
    if len(dataset) == 0:
        logger.log_info("WARNING: Dataset is empty. Check data_dir.")
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    model = ST_GCN(in_channels=3, hidden_channels=64, out_channels=128, num_joints=25)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Simulated data augmentation for contrastive learning (e.g. slight jitter)
            batch_aug1 = batch + torch.randn_like(batch) * 0.01
            batch_aug2 = batch + torch.randn_like(batch) * 0.01
            
            z1 = model(batch_aug1)
            z2 = model(batch_aug2)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(len(dataloader), 1)
        logger.log_epoch(epoch, avg_loss)
        
        logger.save_model(model, optimizer, epoch, is_best=(epoch==args.epochs-1))
        
    logger.plot_losses()
    logger.log_info("Training complete.")

if __name__ == "__main__":
    main()
