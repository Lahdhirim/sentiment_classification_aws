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

def plot_training_and_validation_losses(train_losses: list, val_losses: list, save_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(Fore.MAGENTA + f"Graph saved at {save_path}." + Style.RESET_ALL)
    return None

def clean_checkpoints(train_dir: str) -> None:
    checkpoint_paths = Path(train_dir).glob("checkpoint-*")
    for path in checkpoint_paths:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    print(Fore.MAGENTA + f"Old checkpoints in {train_dir} removed." + Style.RESET_ALL)