import torch.nn as nn
import torch.nn.functional as F


class CNN4Conv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # --- Convolutional layers ---
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # Input: RGB image (3 channels), Output: 32 feature maps, Kernel: 3x3, Padding=1 keeps size
        self.conv2 = nn.Conv2d(32, 32, 3)             # Input: 32 maps, Output: 32 maps, Kernel: 3x3, no padding (reduces size)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Input: 32 maps, Output: 64 maps, Kernel: 3x3, Padding=1
        self.conv4 = nn.Conv2d(64, 64, 3)             # Input: 64 maps, Output: 64 maps, Kernel: 3x3, no padding

        # --- Pooling layer ---
        self.pool = nn.MaxPool2d(2)  # 2x2 Max Pooling, reduces height and width by half

        # --- Dropout layers ---
        self.dropout25 = nn.Dropout(0.25)  # Randomly zero 25% of inputs during training
        self.dropout50 = nn.Dropout(0.5)   # Randomly zero 50% of inputs during training

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(64 * 14 * 14, 512)  # Flattened features (64 channels * 14x14 after conv+pool) → 512 neurons
        self.fc2 = nn.Linear(512, num_classes)   # Final layer: 512 → number of classes

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))  # Apply convolution + ReLU activation
        x = F.relu(self.conv2(x))  # Apply second convolution + ReLU
        x = self.pool(x)  # Max pooling to reduce spatial size
        x = self.dropout25(x)  # Dropout for regularization

        # Second conv block
        x = F.relu(self.conv3(x))  # Third convolution + ReLU
        x = F.relu(self.conv4(x))  # Fourth convolution + ReLU
        x = self.pool(x)  # Max pooling again
        x = self.dropout25(x)  # Dropout

        # Flatten the feature maps to feed into fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 64*14*14)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.dropout50(x)  # Dropout 50%
        x = self.fc2(x)  # FC2 outputs raw logits for each class

        return x  # Return class logits
