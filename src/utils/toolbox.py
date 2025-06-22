import pandas as pd
from colorama import Fore, Style
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

def load_csv_data(data_source: str) -> pd.DataFrame:
    try :
        data = pd.read_csv(data_source)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(
            Fore.RED + f"Could not find the CSV file at {data_source}. Please check the path/URL and try again." + Style.RESET_ALL)

def plot_training_and_validation_curves(train_losses: list, 
                                        val_losses: list,
                                        val_metrics: list,
                                        save_path: str) -> None:
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # loss plot
    ax1.plot(train_losses, label="Train Loss", color='blue')
    ax1.plot(val_losses, label="Validation Loss", color='orange')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # metrics plot
    ax2.plot(val_metrics, label="Validation Accuracy", color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(Fore.MAGENTA + f"Graph saved at {save_path}." + Style.RESET_ALL)

def clean_checkpoints(train_dir: str) -> None:
    checkpoint_paths = Path(train_dir).glob("checkpoint-*")
    for path in checkpoint_paths:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    print(Fore.MAGENTA + f"Old checkpoints in {train_dir} removed." + Style.RESET_ALL)