import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        error_const = torch.erf(x / math.sqrt(2.0))
        x = x * 0.5 * (1.0 + error_const)
        return x