import torch.nn as nn
import torch.nn.functional as F

# 1. Simple CNN Model
class CNN(nn.Module):
    """
    A very simple Convolutional Neural Network:
    - 2 convolutional layers
    - 2 pooling layers
    - 3 fully connected (dense) layers
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # 5x5 kernel, 6 filters
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 5x5 kernel, 16 filters
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flattening for dense layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = F.relu(self.fc2(x))  # Fully connected 2
        x = self.fc3(x)  # Fully connected 3 (output layer)
        return x

# 2. Improved CNN Model
class ImprovedCNN(nn.Module):
    """
    An improved CNN with:
    - 3 convolutional layers
    - Increased number of filters
    - Max pooling for down-sampling
    - Fully connected layers for classification
    """
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32 filters, 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 spatial dimension
            nn.Conv2d(32, 64, 3, padding=1),  # 64 filters
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),  # Reduce filters back to 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8 spatial dimension
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),  # Flatten input to 64-dim vector
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten for dense layers
        x = self.classifier(x)
        return x

# 3. AsimNet Model
class AsimNet(nn.Module):
    """
    A deeper CNN with:
    - 4 convolutional layers
    - Increased depth and filters
    - Max pooling for down-sampling
    - Fully connected layers for classification
    """
    def __init__(self, num_classes=10):
        super(AsimNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Flattening for dense layers
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # First pooling
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # Second pooling
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x

# 4. BN_AsimNet Model
class BN_AsimNet(nn.Module):
    """
    A deeper CNN with:
    - Batch Normalization after each convolution
    - Dropout to prevent overfitting
    - Deeper structure with 4 convolutional layers
    """
    def __init__(self, num_classes=10):
        super(BN_AsimNet, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 256 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x
