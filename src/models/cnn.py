from __future__ import annotations

import torch
import torch.nn as nn
from src.utils.config import Config

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
# https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu
# https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#avgpool1d

#
# Building Block
#


class ConvBlock(nn.Module):
    """
    Single conv block: Conv1D -> BatchNorm1D -> ReLU -> Dropout

    padding=kernel_size // 2 preserves the time dimension across all blocks,
    so depth can be changed freely via config without breaking shapes.

    bias=False because BatchNorm's learnable shift parameter (beta) subsumes
    the conv bias, making it redundant.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


#
# Model
#


class CNN(nn.Module):
    """
    1D CNN for multivariate time series gas classification.

    Architecture
        Input [B, input_channels, window_size]
            N x ConvBlock (expanding channel depth)
            Global Average Pooling  [B, hidden_channels[-1]]
            Linear classifier       [B, num_classes]

    Global Average Pooling collapses the time dimension so the classifier
    is decoupled from window_size. Changing data.window_size in base.yaml
    does not require any model changes.

    Parameters
        input_channels : int
            Number of input feature channels. Must match
            num_sensors * features_per_sensor (128 by default).
        num_classes : int
            Number of output classes (6 for GAS_MAP).
        hidden_channels : list[int]
            Channel depth for each conv block. Length determines the number
            of blocks. Example: [64, 128, 256] produces 3 blocks.
        kernel_size : int
            Convolutional kernel size. Must be odd to keep same-padding exact.
        dropout : float
            Dropout probability applied after each ReLU.
    """

    def __init__(
        self,
        input_channels: int,  # 128 base.yaml
        num_classes: int,  # 6 base.yaml
        hidden_channels: list[int],  # [64, 128, 256] cnn.yaml
        kernel_size: int,  # 3 cnn.yaml
        dropout: float,  # 0.3 base.yaml
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for same-padding to be exact,"
                f"got {kernel_size}."
            )

        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one value.")

        # dynamically build conv blocks from hidden_channels list
        channel_sizes = [input_channels] + hidden_channels

        self.conv_blocks = nn.Sequential(
            *[
                ConvBlock(channel_sizes[i], channel_sizes[i + 1], kernel_size, dropout)
                for i in range(len(hidden_channels))
            ]
        )

        # Global Avg Pooling collapses time dimenstion: [B, C, W] -> [B, C]
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Linear(hidden_channels[-1], num_classes)

        self._init_weights()

    #
    # Construction
    #

    @classmethod
    def from_config(cls, cfg: Config) -> CNN:
        """
        Instantiate a CNN from a merged Config object.

        Expects the following keys to be present (base.yaml + cnn.yaml):
            model.input_channels
            model.num_classes
            model.hidden_channels
            model.kernel_size
            model.dropout
        """
        return cls(
            input_channels=cfg.model.input_channels,
            num_classes=cfg.model.num_classes,
            hidden_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )

    #
    # Weight Initialization
    #

    def _init_weights(self) -> None:
        """Kaiming uniform init for conv layers, zeros for BN bias."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
            x : torch.Tensor of shape [B, input_channels, window_size]

        Returns
            logits : torch.Tensor of shape [B, num_classes]
        """
        x = self.conv_blocks(x)  # [B, hidden_channels[-1], W]
        x = self.gap(x)  # [B, hidden_channels[-1], 1]
        x = x.squeeze(-1)  # [B, hidden_channels[-1]]
        return self.classifier(x)  # [B, num_classes]

    #
    # Utilities
    #

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Return total parameter count.

        Parameters
            trainable_only : bool
                If True (default), count only parameters that receive gradients.
                If False, count all parameters including frozen ones.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"CNN("
            f"input_channels={self.conv_blocks[0].block[0].in_channels}, "
            f"hidden_channels={[b.block[0].out_channels for b in self.conv_blocks]}, "
            f"num_classes={self.classifier.out_features}, "
            f"params={self.num_parameters():,}"
            f")"
        )
