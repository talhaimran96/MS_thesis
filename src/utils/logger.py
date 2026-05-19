import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch

class ExperimentLogger:
    def __init__(self, experiment_name: str, base_dir: str = "Results"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(base_dir, f"{self.experiment_name}_{self.timestamp}")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join("Data", "models", self.experiment_name), exist_ok=True)
        
        self.log_file = os.path.join(self.run_dir, "training.log")
        self.metrics_file = os.path.join(self.run_dir, "metrics.json")
        
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": []
        }
        
        # Setup standard Python logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Experiment Logger for: {self.experiment_name}")

    def log_info(self, message: str):
        self.logger.info(message)

    def log_hyperparams(self, params: dict):
        path = os.path.join(self.run_dir, "hyperparams.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        self.logger.info(f"Hyperparameters saved to {path}")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None, val_acc: float = None):
        self.metrics["epochs"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss)
        if val_acc is not None:
            self.metrics["val_acc"].append(val_acc)
            
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        if val_acc is not None:
            msg += f" | Val Acc: {val_acc:.4f}"
        self.logger.info(msg)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["epochs"], self.metrics["train_loss"], label="Train Loss")
        if self.metrics["val_loss"]:
            plt.plot(self.metrics["epochs"], self.metrics["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve - {self.experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "loss_curve.png"))
        plt.close()

    def save_model(self, model, optimizer, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        save_dir = os.path.join("Data", "models", self.experiment_name)
        last_path = os.path.join(save_dir, "model_last.pth")
        torch.save(state, last_path)
        self.logger.info(f"Saved last model to {last_path}")
        
        if is_best:
            best_path = os.path.join(save_dir, "model_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saved new best model to {best_path}")
