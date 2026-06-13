import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rgb_dataset import GMDCSA24VideoDataset
from src.models.videomae import VideoMAE, VideoMAEForClassification
from src.models.baseline_3dcnn import ResNet3DBaseline
from src.utils.logger import ExperimentLogger

def get_args():
    parser = argparse.ArgumentParser('Experiment A Downstream Fine-tuning')
    parser.add_argument('--data_path', default='Data/raw/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master', type=str)
    parser.add_argument('--model_type', default='videomae', choices=['videomae', 'baseline_3dcnn'])
    parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pre-trained weights')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--freeze_epochs', default=5, type=int, help='Number of epochs to freeze backbone (linear probing)')
    parser.add_argument('--split', default=1.0, type=float, help='Data ablation split ratio (e.g., 0.1 for 10%)')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation (random flip + temporal jitter) during training')
    return parser.parse_args()

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(videos)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, precision, recall, f1, all_preds, all_labels

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Starting K-Fold Cross Validation for {args.model_type} with split={args.split}")
    
    # K-Fold Loop
    for test_subject in range(1, 5):
        logger = ExperimentLogger(experiment_name=f"exp-A_{args.model_type}_subj{test_subject}_split{args.split}")
        logger.log_hyperparams(vars(args))
        logger.log_info(f"--- Fold for Test Subject {test_subject} ---")
        
        # Load Data
        dataset_train = GMDCSA24VideoDataset(args.data_path, test_subject=test_subject, split='train', split_ratio=args.split, augment=args.augment)
        dataset_val = GMDCSA24VideoDataset(args.data_path, test_subject=test_subject, split='val')
        
        if len(dataset_train) == 0 or len(dataset_val) == 0:
            logger.log_info(f"Skipping Subject {test_subject} due to missing data.")
            continue
            
        data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Build Model
        if args.model_type == 'videomae':
            base_model = VideoMAE(img_size=224, patch_size=16, in_chans=3, num_frames=16, tube_size=2, embed_dim=768, depth=12, num_heads=12)
            if args.pretrained_weights and os.path.exists(args.pretrained_weights):
                checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
                base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.log_info(f"Loaded pretrained weights from {args.pretrained_weights}")
            model = VideoMAEForClassification(base_model, num_classes=2)
        else:
            model = ResNet3DBaseline(num_classes=2, pretrained=True)
            
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        # For linear probing, we only pass fc parameters initially
        if args.freeze_epochs > 0:
            if args.model_type == 'videomae':
                for param in model.videomae.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)
            else:
                for param in model.model.parameters():
                    param.requires_grad = False
                for param in model.model.fc.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.model.fc.parameters(), lr=args.lr)
            logger.log_info(f"Freezing backbone for {args.freeze_epochs} epochs.")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            
        best_val_f1 = 0.0
        
        for epoch in range(args.epochs):
            # Check if we need to unfreeze
            if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
                logger.log_info("Unfreezing backbone for end-to-end training.")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 10.0) # Lower LR for fine-tuning
                
            model.train()
            total_loss = 0
            
            pbar = tqdm(data_loader_train, desc=f"Subj {test_subject} - Epoch {epoch}/{args.epochs}")
            for batch_idx, (videos, labels) in enumerate(pbar):
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                logits = model(videos)
                loss = criterion(logits, labels)
                
                loss = loss / args.accum_iter
                loss.backward()
                
                if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(data_loader_train)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * args.accum_iter
                pbar.set_postfix({'loss': loss.item() * args.accum_iter})
                
                del videos, labels, logits
                torch.cuda.empty_cache()
                
            avg_train_loss = total_loss / len(data_loader_train)
            
            # Evaluate
            val_acc, val_precision, val_recall, val_f1, all_preds, all_labels = evaluate(model, data_loader_val, device)
            
            logger.log_epoch(epoch, train_loss=avg_train_loss, val_loss=0.0, val_acc=val_acc, 
                             val_precision=val_precision, val_recall=val_recall, val_f1=val_f1)
            logger.plot_losses()
            
            is_best = val_f1 >= best_val_f1
            if is_best:
                best_val_f1 = val_f1
                logger.save_evaluation_results(all_preds, all_labels, class_names=['ADL', 'Fall'])
                
            logger.save_model(model, optimizer, epoch, is_best=is_best)
            
        logger.log_info(f"Finished Subj {test_subject} with Best F1: {best_val_f1:.4f}")

if __name__ == '__main__':
    args = get_args()
    main(args)
