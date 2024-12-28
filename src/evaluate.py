import torch

def validate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test data.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation or test data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        avg_loss (float): Average loss over the validation set.
        accuracy (float): Accuracy on the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Accumulate loss and metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)  # Get the predicted class
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)  # Average loss
    accuracy = 100. * correct / total       # Accuracy percentage

    return avg_loss, accuracy
