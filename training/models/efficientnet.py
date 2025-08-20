#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
from torch.nn import Sequential
from torch.nn.functional import avg_pool1d

from training.models.utils import Conv1dPadding, MBConv1d


class EfficientNet(torch.nn.Module):
    """Efficient net encdoer for 1D time series
    Args:
        input_shape (int): number of input channels
        output_shape (int): dimension of output/embeddings
        kernel_size (int): default kernel size
        activation (str): activation function,
                          only supports "swish" and "relu" for now
        expansion_factor (int): expansion factor for number of channels
                                in the MB modules
        se_ratio (float): squeeze and excitation ratio,
                          if None it skips it
        downsample_factor (int): factor by which time dimension is reduced using strides
                            One of: 1, 2, 4, 8, 16, 32, 64
    Outputs:
        x (torch.Tensor): Output/Embeddings after average pooling in the temporal dim
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        kernel_size: int = 3,
        activation: str = "swish",
        expansion_factor: int = 3,
        se_ratio: float = 0.25,
        output_transform: str = "global_avg_pool",
        downsample_factor: int = 64,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.activation = activation
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio
        self.output_transform = output_transform

        self.layers = Sequential()

        assert downsample_factor in 2 ** np.arange(7)
        ds = int(np.log2(downsample_factor))
        strides = 2 * np.ones(6).astype(int)  # Stride of 2 applied to selected 6 layers
        strides[ds:] = 1

        # Initial Convolution
        filter_factor = self.output_shape / 36

        layer = Conv1dPadding(
            in_channels=self.input_shape,
            out_channels=int(filter_factor * 6),
            kernel_size=self.kernel_size,
            stride=strides[0],
            padding_strategy="same",
            bias=False,
            use_batch_norm=True,
            activation=self.activation,
        )
        self.layers.add_module("Conv1D_Initial", layer)

        # Mobile Inverted Bottlenecks (Inverted Residual Blocks)
        # channels *4
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=int(filter_factor * 4),
            kernel_size=self.kernel_size,
            stride=1,
            expansion_factor=1,
            activation=self.activation,
            se_ratio=self.se_ratio,
        )
        self.layers.add_module("MBConv1d_1", layer)

        common_kwargs = dict(
            expansion_factor=self.expansion_factor,
            activation=self.activation,
            se_ratio=self.se_ratio,
        )

        # channels * 5 (stride=2 first, then stride=1)
        this_layer_kernel_size = self.kernel_size
        this_layer_out_channels = int(filter_factor * 5)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=strides[1],
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_2", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_3", layer)

        # channels * 7 (stride=2 first, then stride=1); kernel_size=2 * kernel_size - 1
        this_layer_kernel_size = 2 * self.kernel_size - 1
        this_layer_out_channels = int(filter_factor * 7)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=strides[2],
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_4", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_5", layer)

        # channels * 9 (stride=2 first, then 2x stride=1); kernel_size=kernel_size
        this_layer_kernel_size = self.kernel_size
        this_layer_out_channels = int(filter_factor * 9)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=strides[3],
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_6", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_7", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_8", layer)

        # channels * 11 (stride=2 first, then 2x stride=1); kernel_size=2*kernel_size-1
        this_layer_kernel_size = 2 * self.kernel_size - 1
        this_layer_out_channels = int(filter_factor * 11)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=strides[4],
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_9", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_10", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_11", layer)

        # channels * 14 (stride=2 first, then 3x stride=1); kernel_size=2*kernel_size-1
        this_layer_kernel_size = 2 * self.kernel_size - 1
        this_layer_out_channels = int(filter_factor * 14)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=strides[5],
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_12", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_13", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_14", layer)

        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_15", layer)

        # channels * 18 (stride=1); kernel_size=2*kernel_size-1
        this_layer_kernel_size = 2 * self.kernel_size - 1
        this_layer_out_channels = int(filter_factor * 18)
        layer = MBConv1d(
            in_channels=layer.out_ch,
            out_channels=this_layer_out_channels,
            kernel_size=this_layer_kernel_size,
            stride=1,
            **common_kwargs,
        )
        self.layers.add_module("MBConv1d_16", layer)

        # Final Expansion (pointwise convolution)
        layer = Conv1dPadding(
            in_channels=layer.out_ch,
            out_channels=int(filter_factor * 36),
            kernel_size=1,
            stride=1,
            padding_strategy="SAME",
            bias=False,
            use_batch_norm=True,
            activation=self.activation,
        )
        self.layers.add_module("Conv1D_Final_Expansion", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        if self.output_transform == "global_avg_pool":
            # Average over the time domain to get a single representation
            # vector for the whole segment:
            x = avg_pool1d(x, kernel_size=int(x.shape[-1]))  # Global average pooling
            x = torch.squeeze(x, dim=2)

        return x
