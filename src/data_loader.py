import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader

def get_data_loaders(batch_size=64, train_ratio=0.8, augment=False, root='./data'):
    """
    Create data loaders for training, validation, and testing with optional data augmentation.

    Parameters:
        batch_size (int): Batch size for data loaders.
        train_ratio (float): Proportion of training data.
        augment (bool): Whether to apply data augmentation.
        root (str): Path to save/load CIFAR-10 dataset.

    Returns:
        trainloader, valloader, testloader (DataLoader): Data loaders for train, validation, and test sets.
    """
    if augment:
        # Data Augmentation with normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        # Basic normalization only
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    # Split training set into training and validation
    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader
