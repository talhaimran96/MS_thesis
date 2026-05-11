import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.velocity_dataset import VelocityMediaPipeDataset
from src.models.velocity_classifier import VelocityClassifier
from src.utils.logger import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description="Train Velocity Classifier (MediaPipe Stream)")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length for velocity vectors')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    hyperparams = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': 1e-3,
        'seq_len': args.seq_len,
        'input_size': 99, # 33 joints * 3 dims (x,y,z)
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 2
    }
    
    logger = ExperimentLogger(branch_name="velocity_classification", 
                              model_type="mediapipe_lstm", 
                              hyperparams=hyperparams,
                              resume=(args.resume is not None))
    
    # --- Data Loading ---
    data_dir = 'Data/raw/kinetics400_5per/train/'
    
    print(f"Loading MediaPipe Video Dataset from {data_dir}...")
    full_dataset = VelocityMediaPipeDataset(data_dir=data_dir, seq_len=hyperparams['seq_len'], binary_mode=True)
    
    if len(full_dataset) == 0:
        print("WARNING: Dataset is empty or path doesn't exist. Using dummy data for testing.")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                return torch.randn(hyperparams['seq_len'], hyperparams['input_size']), torch.randint(0, 2, (1,)).item()
        full_dataset = DummyDataset()
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=0, drop_last=True)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # --- Model Initialization ---
    print("Initializing Velocity Classifier...")
    model = VelocityClassifier(input_size=hyperparams['input_size'], 
                               hidden_size=hyperparams['hidden_size'], 
                               num_layers=hyperparams['num_layers'], 
                               num_classes=hyperparams['num_classes'])
    model.to(device)
    
    if hasattr(full_dataset, 'samples'):
        num_falls = sum(1 for _, label in full_dataset.samples if label == 1)
        num_adl = len(full_dataset.samples) - num_falls
        if num_falls > 0:
            weight_0 = 1.0
            weight_1 = num_adl / num_falls
            class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
        else:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
        
    print(f"Class Weights: ADL={class_weights[0].item():.2f}, Fall={class_weights[1].item():.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=1e-6)
    
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()
                
        logger.train_losses = logger.train_losses[:start_epoch]
        logger.val_losses = logger.val_losses[:start_epoch]
        print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    
    print("Starting Training...")
    for epoch in range(start_epoch, hyperparams['epochs']):
        # Train Loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Train]")
        for batch_idx, (sequences, labels) in enumerate(pbar_train):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
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
            for batch_idx, (sequences, labels) in enumerate(pbar_val):
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
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
        
        scheduler.step()
        logger.log_epoch(epoch + 1, avg_train_loss, avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, logger.get_best_model_path())
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
            
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, logger.get_last_model_path())
    print("Training loop complete. Models and logs saved.")

    # --- Final Evaluation (Confusion Matrix) ---
    print("Generating Confusion Matrix on Validation Set with Best Model...")
    best_checkpoint = torch.load(logger.get_best_model_path(), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating Best Model"):
            sequences = sequences.to(device)
            outputs = model(sequences)
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
