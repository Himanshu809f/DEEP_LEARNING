# model.py
import torch
import torch.nn as nn
from torchvision import models

class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # use torchvision resnet18 but adapt first conv for CIFAR (3x32x32)
        self.model = models.resnet18(pretrained=pretrained)
        # Replace the first conv to kernel_size=3, stride=1, padding=1 if desired
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # remove the 7x7 maxpool for small images
        # Replace fc
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
