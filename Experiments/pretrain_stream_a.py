import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rgb_dataset import KineticsVideoDataset
from src.models.videomae import VideoMAE, TubeMaskingGenerator
from src.utils.logger import ExperimentLogger

def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--mask_ratio', default=0.9, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--data_path', default='Data/raw/kinetics400_5per/train', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    return parser.parse_args()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = ExperimentLogger(experiment_name="exp-A-rgb-baseline_videomae")
    logger.log_hyperparams(vars(args))

    # Dataset
    logger.log_info(f"Loading data from {args.data_path}")
    dataset_train = KineticsVideoDataset(args.data_path, num_frames=16, frame_size=224, split='train')
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Model
    model = VideoMAE(img_size=224, patch_size=16, in_chans=3, num_frames=16, tube_size=2, embed_dim=768, depth=12, num_heads=12)
    model.to(device)

    # Tube Masking Generator
    # input_size is (T, H, W) in terms of patches. For 16 frames, tube_size 2 => T=8. For 224, patch 16 => 14x14
    window_size = (16 // 2, 224 // 16, 224 // 16)
    mask_generator = TubeMaskingGenerator(window_size, args.mask_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.log_info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.log_info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logger.log_info(f"No checkpoint found at '{args.resume}'")

    logger.log_info(f"Start pretraining for {args.epochs} epochs")
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (videos, _) in enumerate(pbar):
            videos = videos.to(device, non_blocking=True)
            
            # Generate mask
            # mask shape is (B, N)
            mask = torch.tensor([mask_generator() for _ in range(videos.size(0))]).to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model(videos, mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(data_loader_train)
        
        logger.log_epoch(epoch, train_loss=avg_loss)
        logger.plot_losses()
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        logger.save_model(model, optimizer, epoch, is_best=is_best)

if __name__ == '__main__':
    args = get_args()
    main(args)
