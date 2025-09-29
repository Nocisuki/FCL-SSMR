import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntheticCNN(nn.Module):
    def __init__(self, num_channels=3, args=None):
        super(SyntheticCNN, self).__init__()
        assert args is not None, "you should pass args to SyntheticCNN"

        if 'mnist' in args["dataset"]:
            num_channels = 1

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 128

    def forward(self, x):
        fmaps = []

        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        fmaps.append(x1)

        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        fmaps.append(x2)

        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        fmaps.append(x3)

        features = torch.flatten(self.global_pool(x3), 1)

        return {
            "fmaps": fmaps,
            "features": features
        }

    @property
    def last_conv(self):
        return self.conv3
