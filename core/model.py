"""
Base ML model that will receive input configurations from UI
"""

import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
