import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rgb_dataset import OOPSVideoDataset
from src.models.videomae import VideoMAE, VideoMAEForClassification
from src.models.baseline_3dcnn import ResNet3DBaseline
from src.utils.logger import ExperimentLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_args():
    parser = argparse.ArgumentParser('Experiment A OOPS-Fall Testing')
    parser.add_argument('--dataset', default='oops_fall', type=str, help='Dataset to test on (default: oops_fall)')
    parser.add_argument('--data_path', default='Data/raw/oops_dataset', type=str)
    parser.add_argument('--model_type', default='videomae', choices=['videomae', 'baseline_3dcnn'])
    parser.add_argument('--weights', default='', type=str, required=True, help='Path to fine-tuned weights')
    parser.add_argument('--batch_size', default=4, type=int)
    return parser.parse_args()

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc="Evaluating on OOPS-Fall"):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(videos)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, precision, recall, f1, all_preds, all_labels

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = ExperimentLogger(experiment_name=f"test-A_{args.model_type}_oops")
    logger.log_hyperparams(vars(args))
    logger.log_info(f"Starting OOPS-Fall Evaluation for {args.model_type}")
    
    # Load Data
    dataset_test = OOPSVideoDataset(args.data_path)
    
    if len(dataset_test) == 0:
        logger.log_info("Testing dataset is empty. Exiting.")
        return
        
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Build Model
    if args.model_type == 'videomae':
        base_model = VideoMAE(img_size=224, patch_size=16, in_chans=3, num_frames=16, tube_size=2, embed_dim=768, depth=12, num_heads=12)
        model = VideoMAEForClassification(base_model, num_classes=2)
    else:
        model = ResNet3DBaseline(num_classes=2, pretrained=False)
        
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.log_info(f"Successfully loaded fine-tuned weights from {args.weights}")
    else:
        logger.log_info(f"ERROR: Weights not found at {args.weights}. Exiting.")
        return
        
    model.to(device)
    
    # Evaluate
    acc, precision, recall, f1, all_preds, all_labels = evaluate(model, data_loader_test, device)
    
    logger.log_info("--- OOPS-Fall Testing Results ---")
    logger.log_info(f"Accuracy:  {acc:.4f}")
    logger.log_info(f"Precision: {precision:.4f}")
    logger.log_info(f"Recall:    {recall:.4f}")
    logger.log_info(f"F1-Score:  {f1:.4f}")
    
    logger.save_evaluation_results(all_preds, all_labels, class_names=['ADL', 'Fall'])
    logger.log_info("Testing Complete.")

if __name__ == '__main__':
    args = get_args()
    main(args)
