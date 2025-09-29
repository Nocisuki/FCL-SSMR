import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, num_channels=3, args=None):
        super(VGG, self).__init__()
        assert args is not None, "you should pass args to VGG11"

        if "mnist" in args["dataset"]:
            num_channels = 1

        # VGG-11
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /4

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /8

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /16

            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        self.out_dim = 512

    def forward(self, x):
        fmaps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                fmaps.append(x)

        features = torch.flatten(x, 1)  # [bs, 512]

        return {
            "fmaps": fmaps,
            "features": features
        }

    @property
    def last_conv(self):
        for layer in reversed(self.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No Conv2d layer found in VGG11")
