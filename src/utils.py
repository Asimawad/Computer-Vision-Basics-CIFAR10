import matplotlib.pyplot as plt
import torch
import time

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, metric_type="Validation"):
    """
    Plot training and validation loss/accuracy.

    Parameters:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
        train_accuracies (list): List of training accuracies over epochs.
        val_accuracies (list): List of validation accuracies over epochs.
        metric_type (str): Metric label for validation/test metrics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label=f'{metric_type} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()

    # Plot Accuracy
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label=f'{metric_type} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def setup_device():
    """
    Set up device for training.

    Returns:
        device (torch.device): GPU if available, else CPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

class Timer:
    """
    Utility class to track execution time of code blocks.
    """
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed_time = time.time() - self.start_time
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
        return elapsed_time
