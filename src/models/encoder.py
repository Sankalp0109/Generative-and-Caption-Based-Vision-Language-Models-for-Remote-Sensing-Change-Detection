"""Encoder architectures for change image pairs."""

import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """
    Two-stream CNN encoder with before/after/difference fusion.

    Input:  (batch, 2, 3, H, W)
    Output: (batch, out_dim)
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, out_dim: int = 512):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)

        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)

        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim * 4 * 3, out_dim)

    def encode_single_stream(self, image):
        features = self.relu(self.bn1(self.conv1(image)))
        features = self.relu(self.bn2(self.conv2(features)))
        features = self.relu(self.bn3(self.conv3(features)))
        features = self.adaptive_pool(features)
        return features.view(features.size(0), -1)

    def forward(self, images):
        before_feat = self.encode_single_stream(images[:, 0])
        after_feat = self.encode_single_stream(images[:, 1])
        diff_feat = torch.abs(before_feat - after_feat)
        fused = torch.cat([before_feat, after_feat, diff_feat], dim=1)
        return self.fc(fused)
