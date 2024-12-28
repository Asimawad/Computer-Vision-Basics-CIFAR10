import torch
import torch.optim as optim
import torch.nn as nn
import csv
from src.data_loader import get_data_loaders
from src.models import CNN, ImprovedCNN, AsimNet, BN_AsimNet
from src.train import train_model
from src.evaluate import validate_model
from src.utils import plot_metrics, setup_device

def train_and_evaluate(model, model_name, trainloader, valloader, testloader, device, num_epochs=10):
    """
    Train and evaluate a given model.

    Parameters:
        model (torch.nn.Module): The model to train.
        model_name (str): Name of the model (for saving results).
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        testloader (DataLoader): Test data loader.
        device (torch.device): Device to run the training on.
        num_epochs (int): Number of epochs to train the model.
    """
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_accuracy = train_model(model, trainloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validate
        val_loss, val_accuracy = validate_model(model, valloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Evaluate on test set
    test_loss, test_accuracy = validate_model(model, testloader, criterion, device)
    print(f"\n[Test] Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), f'./saved_models/{model_name}.pth')
    print(f"Model {model_name} saved.")

    # Save metrics to CSV
    with open(f'./results/{model_name}_metrics.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
        for i in range(num_epochs):
            writer.writerow([i + 1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])
    print(f"Metrics for {model_name} saved.")

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def main():
    # Set up device
    device = setup_device()

    # Fetch dataset
    trainloader, valloader, testloader = get_data_loaders(batch_size=64, augment=True)

    # Define models to train
    models = {
        "SimpleCNN": CNN(num_classes=10),
        "ImprovedCNN": ImprovedCNN(num_classes=10),
        "AsimNet": AsimNet(num_classes=10),
        "BN_AsimNet": BN_AsimNet(num_classes=10)
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...\n")
        model = model.to(device)
        train_and_evaluate(model, model_name, trainloader, valloader, testloader, device, num_epochs=10)

if __name__ == "__main__":
    main()
