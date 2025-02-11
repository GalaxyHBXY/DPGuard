import torch.nn as nn
from torchvision import models

class ResNetBinary(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetBinary, self).__init__()
        self.resnet = models.resnet101(weights='DEFAULT')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x).squeeze(1)
