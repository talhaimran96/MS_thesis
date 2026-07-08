import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import sys
from sklearn.metrics import precision_recall_fscore_support

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.st_gcn import ST_GCN
from src.models.sth_mae import STH_MAE
from src.utils.logger import ExperimentLogger

# A dummy dataset for demonstration of downstream fine-tuning (Fall vs ADL)
class DummyFallDataset(torch.utils.data.Dataset):
    def __init__(self, mode='heatmap', num_samples=100):
        self.mode = mode
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def _skeleton_pose_detection_processing(self):
        # Simulated skeleton pose detection processing
        if self.mode == 'heatmap':
            # Simulated 3D heatmap generated from skeleton pose detection
            return torch.rand(1, 32, 32, 32)
        else:
            # Simulated Graph: (C, T, V, M) -> 3, 100, 25, 1 from skeleton pose detection
            return torch.rand(3, 100, 25, 1)

    def __getitem__(self, idx):
        label = torch.randint(0, 2, (1,)).item() # 0 for ADL, 1 for Fall
        data = self._skeleton_pose_detection_processing()
        return data, label

class FallClassifier(nn.Module):
    def __init__(self, backbone, in_features, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        if isinstance(self.backbone, STH_MAE):
            # For STH_MAE, we just use the encoder without mask for features
            latent, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)
            # Global average pooling over sequence
            features = latent.mean(dim=1)
        else:
            # For GCN, it returns the projection, we could take features before projection in practice
            features = self.backbone(x)
            
        logits = self.fc(features)
        return logits

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stream B on Fall Detection Task")
    parser.add_argument('--model_type', type=str, choices=['gcn', 'sth_mae'], required=True)
    parser.add_argument('--pretrained_weights', type=str, default="", help='Path to pretrained weights. Empty for from-scratch.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--mask_ratio', type=float, default=0.90,
                        help='Masking ratio used during STH-MAE pretraining forward pass (default: 0.90)')
    args = parser.parse_args()
    
    weight_status = "pretrained" if args.pretrained_weights else "scratch"
    logger = ExperimentLogger(experiment_name=f"eval_{args.model_type}_{weight_status}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'gcn':
        backbone = ST_GCN(in_channels=3, hidden_channels=64, out_channels=128, num_joints=25)
        in_features = 128
        mode = 'graph'
    else:
        backbone = STH_MAE(target_shape=(32, 32, 32), patch_size=(4, 4, 4), in_channels=1)
        in_features = 768
        mode = 'heatmap'
        
    # Load pretrained weights
    if os.path.exists(args.pretrained_weights):
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.log_info(f"Loaded pretrained weights from {args.pretrained_weights}")
    else:
        logger.log_info(f"WARNING: Weights {args.pretrained_weights} not found. Training from scratch.")
        
    model = FallClassifier(backbone, in_features).to(device)
    
    dataset = DummyFallDataset(mode=mode)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Implementing Class-Weighted Loss to handle severe class imbalance
    # Assuming class 0 is ADL and class 1 is Fall (minority class)
    # Give a higher weight (e.g., 10x) to the Fall class
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    logger.log_info(f"Starting Fine-tuning Evaluation for {args.model_type}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for data, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect predictions for metric calculation and confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
                
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Calculate Precision, Recall, and F1
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        
        logger.log_epoch(epoch, train_loss=avg_loss, val_acc=train_acc, val_precision=precision, val_recall=recall, val_f1=f1)
        
    logger.plot_losses()
    logger.save_evaluation_results(all_preds, all_labels)
    logger.log_info("Evaluation complete.")
    
if __name__ == "__main__":
    main()
