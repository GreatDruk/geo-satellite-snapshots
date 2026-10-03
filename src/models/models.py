"""Model definitions."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple convolutional neural network."""

    def __init__(self, num_classes: int = 20, activation: str = "relu", input_channels: int = 3):
        super().__init__()
        self.activation = nn.ModuleDict({
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        })

        activation = self.activation.get(activation, nn.ReLU())
        self.layers = nn.ModuleList([
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            activation,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        for layer in self.layers:
            x = layer(x)
        return x
