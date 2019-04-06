import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNet


class MTPNet(nn.Module):
    def __init__(self, input_channels, num_classes, embedding_size):
        super(MTPNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_channels = input_channels
        self.net = PointNet(self.input_channels)
        self.fc1 = nn.Conv1d(128, self.num_classes, 1)
        self.fc2 = nn.Conv1d(128, self.embedding_size, 1)

    def forward(self, x):
        x = self.net(x)
        logits = self.fc1(x)
        logits = logits.transpose(2, 1)
        logits = torch.log_softmax(logits, dim=-1)
        embedded = self.fc2(x)
        embedded = embedded.transpose(2, 1)
        return logits, embedded
