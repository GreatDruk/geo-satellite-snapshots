"""Parameter-free random instance-statistics mixing (Zhou et al., ICLR 2021)."""
import math
import random

import torch
from torch import nn


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        for name, value in (("p", p), ("alpha", alpha), ("eps", eps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"MixStyle {name} must be finite")
        if not 0 <= p <= 1 or alpha <= 0 or eps <= 0:
            raise ValueError("MixStyle requires p in [0,1], positive alpha and eps")
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, features):
        # Eval returns the same tensor and consumes NO random numbers.
        if not self.training or self.p == 0 or features.shape[0] < 2:
            return features
        if random.random() >= self.p:
            return features
        if features.ndim != 4:
            raise ValueError("MixStyle expects [B,C,H,W]")
        stats = features.float()
        mean = stats.mean((2, 3), keepdim=True).detach()
        variance = stats.var((2, 3), keepdim=True, unbiased=features.shape[2] * features.shape[3] > 1)
        std = (variance + self.eps).sqrt().detach()
        normalized = (stats - mean) / std
        # Sampling on CPU follows the reference implementation; to(device) is
        # explicit so this works on CPU and CUDA and with the shared seed code.
        coefficient = torch.distributions.Beta(self.alpha, self.alpha).sample((len(features), 1, 1, 1)).to(features.device)
        permutation = torch.randperm(len(features), device=features.device)
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[permutation]
        mixed_std = coefficient * std + (1 - coefficient) * std[permutation]
        return (normalized * mixed_std + mixed_mean).to(features.dtype)
