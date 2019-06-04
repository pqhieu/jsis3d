import torch
import torch.nn as nn


class NLLLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.nll = nn.NLLLoss(weight, reduction=reduction)

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.view(batch_size * num_points, -1)
        y = y.view(batch_size * num_points)
        loss = self.nll(x, y)
        return loss
