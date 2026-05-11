import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.ntu_skeleton_dataset import NTUSkeletonDataset
from src.models.velocity_classifier import VelocityClassifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Matching the hyperparams from the training script
    seq_len = 60
    input_size = 75
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 16
    
    # Path to the best saved model
    model_path = 'Data/models/velocity_classification/skeleton_lstm_best.pth'
    results_dir = 'Results/velocity_classification_skeleton_lstm'
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}. Make sure training has completed.")
        return

    # --- Data Loading ---
    data_dir = 'Data/raw/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/'
    print(f"Loading NTU Skeleton Dataset from {data_dir}...")
    
    full_dataset = NTUSkeletonDataset(data_dir=data_dir, seq_len=seq_len)
    
    if len(full_dataset) == 0:
        print("WARNING: Dataset is empty or path doesn't exist. Using dummy data for testing.")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                return torch.randn(seq_len, input_size), torch.randint(0, 2, (1,)).item()
        full_dataset = DummyDataset()
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    # --- Model Initialization ---
    print(f"Loading model from {model_path}...")
    model = VelocityClassifier(input_size=input_size, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               num_classes=num_classes)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Final Evaluation (Confusion Matrix) ---
    print("Generating Confusion Matrix on Validation Set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating Model"):
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADL", "Fall"])
        disp.plot(cmap=plt.cm.Blues)
        
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion Matrix successfully saved to: {cm_path}")
    except ImportError:
        print("scikit-learn or matplotlib not installed, skipping confusion matrix generation.")

if __name__ == '__main__':
    main()
