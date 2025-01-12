import torch
import torch.nn as nn

class CustomCNN(nn.Module):

    def __init__(self, num_classes, num_channels):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,padding=3,stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc= nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.fc(x)
        
        return logits