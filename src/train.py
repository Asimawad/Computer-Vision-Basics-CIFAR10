
import torch
import torch.nn as nn
import torch.optim as optim

import torch

def train_model(model, trainloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters:
        model (torch.nn.Module): The model to train.
        trainloader (DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on.

    Returns:
        avg_loss (float): Average loss for the epoch.
        accuracy (float): Training accuracy for the epoch.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(trainloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def validate_model(model, validation_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    validation_accuracy = 100. * correct / total
    print(f"Validation Accuracy: {validation_accuracy}%")
