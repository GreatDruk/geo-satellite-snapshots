"""Avoid CUDA adaptive-average-pool backward when an equivalent regular pool exists."""
from torch import nn
from torch.nn import functional as F
import torch


class GlobalAverageMaxPool2d(nn.Module):
    """Concatenate spatial mean and max; one backbone forward, twice the features."""
    def forward(self, x):
        return torch.cat([x.mean(dim=(-2,-1),keepdim=True), x.amax(dim=(-2,-1),keepdim=True)],dim=1)


class CompatibleAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, x):
        output = self.output_size
        output = (output, output) if isinstance(output, int) else tuple(output)
        height, width = x.shape[-2:]
        out_height, out_width = (height if output[0] is None else output[0],
                                 width if output[1] is None else output[1])
        if (out_height, out_width) == (1, 1):
            return x.mean(dim=(-2, -1), keepdim=True)
        if height % out_height == 0 and width % out_width == 0:
            kernel = (height // out_height, width // out_width)
            return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
        # Nondivisible adaptive bins overlap; do not silently change their definition.
        return super().forward(x)


def replace_adaptive_pools(module):
    for name, child in module.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, CompatibleAdaptiveAvgPool2d(child.output_size))
        else:
            replace_adaptive_pools(child)
