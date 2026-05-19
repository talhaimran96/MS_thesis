import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

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
        
    def __getitem__(self, idx):
        label = torch.randint(0, 2, (1,)).item() # 0 for ADL, 1 for Fall
        if self.mode == 'heatmap':
            data = torch.randn(1, 32, 32, 32)
        else:
            # Graph: (C, T, V, M) -> 3, 100, 25, 1
            data = torch.randn(3, 100, 25, 1)
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
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    logger = ExperimentLogger(experiment_name=f"eval_{args.model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'gcn':
        backbone = ST_GCN(in_channels=3, hidden_channels=64, out_channels=128, num_joints=25)
        in_features = 128
        mode = 'graph'
    else:
        backbone = STH_MAE(target_shape=(32, 32, 32), patch_size=(4, 4, 4), in_channels=1, embed_dim=256)
        in_features = 256
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
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    logger.log_info(f"Starting Fine-tuning Evaluation for {args.model_type}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, labels in dataloader:
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
            
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        logger.log_epoch(epoch, train_loss=avg_loss, val_acc=train_acc)
        
    logger.plot_losses()
    logger.log_info("Evaluation complete.")
    
if __name__ == "__main__":
    main()
