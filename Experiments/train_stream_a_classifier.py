import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.data.kinetics_dataset import KineticsDataset
from src.models.videomae import VideoMAE
from src.utils.logger import ExperimentLogger

class VideoMAEClassifier(nn.Module):
    """
    Classification head attached to the pretrained VideoMAE backbone.
    """
    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        
        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # The embed_dim is accessible via the norm layer's dimension
        embed_dim = self.backbone.norm.weight.shape[0]
        
        # Simple linear classifier head on top of the CLS token
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Pass through the encoder with mask_ratio=0.0 to process all patches
        x_encoded, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)
        
        # Extract the CLS token (the first token in the sequence)
        cls_token = x_encoded[:, 0]
        
        # Classification prediction
        out = self.head(cls_token)
        return out


def main():
    parser = argparse.ArgumentParser(description="Train Stream A Classifier (Fine-Tuning)")
    parser.add_argument('--pretrained', type=str, default='Data/models/stream_a_rgb_baseline/videomae_v2_best.pth', help='Path to pretrained VideoMAE weights')
    parser.add_argument('--resume', type=str, default=None, help='Path to classification checkpoint to resume from')
    parser.add_argument('--num_classes', type=int, default=400, help='Number of classification classes (e.g., 2 for Fall vs ADL, 400 for Kinetics)')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the VideoMAE backbone and only train the classification head')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    hyperparams = {
        'batch_size': 16,
        'epochs': 100,
        'lr': 5e-4, # Smaller learning rate for fine-tuning
        'depth': 4,
        'num_heads': 6,
        'patch_size': 16,
        'tube_size': 2,
        'num_classes': args.num_classes,
        'freeze_backbone': args.freeze_backbone
    }
    
    logger = ExperimentLogger(branch_name="stream_a_classifier", 
                              model_type="videomae_v2_classifier", 
                              hyperparams=hyperparams,
                              resume=(args.resume is not None))
    
    # --- Data Loading ---
    # TODO: Replace KineticsDataset with GMDCSA-24 Dataset for Fall vs ADL classification
    # For now, using Kinetics as a placeholder to ensure the script runs.
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
    
    # --- Model Initialization ---
    print("Initializing Model...")
    backbone = VideoMAE(depth=hyperparams['depth'], num_heads=hyperparams['num_heads'])
    
    # Load pretrained VideoMAE weights if provided
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"Loading pretrained backbone from '{args.pretrained}'...")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        # The checkpoint might contain 'model_state_dict' or just the raw weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load weights into backbone (strict=False because we don't need decoder weights for classification)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded backbone. Missing keys: {len(missing)} (expected if decoder weights are missing), Unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: No pretrained weights found. Training from scratch.")

    model = VideoMAEClassifier(backbone, num_classes=hyperparams['num_classes'], freeze_backbone=hyperparams['freeze_backbone'])
    model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # We might want different learning rates for backbone vs head
    if hyperparams['freeze_backbone']:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=hyperparams['lr'])
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': hyperparams['lr'] * 0.1},
            {'params': model.head.parameters(), 'lr': hyperparams['lr']}
        ])
        
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=1e-6)
    
    start_epoch = 0
    best_val_acc = 0.0

    # Resume from classification checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Catch up scheduler if it wasn't saved in older checkpoints
            for _ in range(start_epoch):
                scheduler.step()
                
        # Ensure logger lists align with the resumed epoch
        logger.train_losses = logger.train_losses[:start_epoch]
        logger.val_losses = logger.val_losses[:start_epoch]
        
        print(f"Loaded classification checkpoint '{args.resume}' (epoch {start_epoch})")
    
    print("Starting Training...")
    for epoch in range(start_epoch, hyperparams['epochs']):
        # Train Loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Train]")
        for batch_idx, (frames, labels) in enumerate(pbar_train):
            frames = frames.permute(0, 2, 1, 3, 4).to(device) # (B, C, T, H, W)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100 * correct_train / total_train:.2f}%"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Val]")
        with torch.no_grad():
            for batch_idx, (frames, labels) in enumerate(pbar_val):
                frames = frames.permute(0, 2, 1, 3, 4).to(device)
                labels = labels.to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100 * correct_val / total_val:.2f}%"})
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        current_lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%)")
        
        # Step the scheduler
        scheduler.step()
        
        # Log to file and plot
        logger.log_epoch(epoch + 1, avg_train_loss, avg_val_loss)
        # Note: We might want to expand ExperimentLogger to handle accuracy too, but for now we log loss.
        
        # Save Best Model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': avg_val_loss
            }, logger.get_best_model_path())
            print(f"--> Saved new best model with Val Acc: {best_val_acc:.2f}%")
            
    # Save Last Model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc
    }, logger.get_last_model_path())
    print("Training loop complete. Models and logs saved.")

    # --- Final Evaluation (Confusion Matrix) ---
    print("Generating Confusion Matrix on Validation Set with Best Model...")
    # Load Best Model for final evaluation
    best_checkpoint = torch.load(logger.get_best_model_path(), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Evaluating Best Model"):
            frames = frames.permute(0, 2, 1, 3, 4).to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        
        cm_path = os.path.join(logger.results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion Matrix saved to {cm_path}")
    except ImportError:
        print("scikit-learn or matplotlib not installed, skipping confusion matrix generation.")

if __name__ == '__main__':
    main()
