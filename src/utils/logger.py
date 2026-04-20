import os
import json
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, branch_name, model_type, hyperparams):
        self.branch_name = branch_name
        self.model_type = model_type
        
        # Define directories
        self.results_dir = os.path.join('Results', f"{branch_name}_{model_type}")
        self.models_dir = os.path.join('Data', 'models', branch_name)
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save hyperparams
        with open(os.path.join(self.results_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
            
        self.train_losses = []
        self.val_losses = []
        
    def log_epoch(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Save intermediate results
        self.save_results()
        self.plot_losses()
        
    def save_results(self):
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': min(self.val_losses) if self.val_losses and self.val_losses[0] is not None else None,
            'best_epoch': self.val_losses.index(min(self.val_losses)) + 1 if self.val_losses and self.val_losses[0] is not None else None
        }
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Train Loss', marker='o')
        if self.val_losses and self.val_losses[0] is not None:
            plt.plot(epochs, self.val_losses, label='Validation Loss', marker='o')
            
        plt.title(f'Training & Validation Loss ({self.model_type})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'loss_graph.png'))
        plt.close()
        
    def get_best_model_path(self):
        return os.path.join(self.models_dir, f"{self.model_type}_best.pth")
        
    def get_last_model_path(self):
        return os.path.join(self.models_dir, f"{self.model_type}_last.pth")
