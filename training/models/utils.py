#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
from torch.nn import Conv1d

#
# blocks
#


class Conv1dPadding(torch.nn.Module):
    """conv1d with padding"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding_strategy="same",
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation=None,
        use_batch_norm=False,
    ):
        super().__init__()

        if activation is not None:
            activation = activation.lower()

        if padding_strategy is not None:
            padding_strategy = padding_strategy.lower()

        assert activation in ["relu", "swish"] or activation is None

        self.padding_strategy = padding_strategy
        self.activation = activation

        if self.padding_strategy == "same":
            padding_val = dilation * (kernel_size - 1) // 2
        elif self.padding_strategy == "valid":
            padding_val = 0
        else:
            raise NotImplementedError(
                f"Padding strategy {self.padding_strategy} not implemented!"
            )
        self.conv1d = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding_val,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.batch_norm = None
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation in ["relu", "swish"]:
            if self.activation == "relu":
                x = torch.nn.functional.relu(x)
            elif self.activation == "swish":
                x = torch.nn.functional.silu(x)
        return x

    @property
    def out_ch(self) -> int:
        return self.conv1d.out_channels


class MBConv1d(torch.nn.Module):
    """
    Mobile Inverted Bottleneck for 1D.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding_strategy="same",
        expansion_factor=3,
        activation="swish",
        se_ratio=0.25,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        if activation is not None:
            activation = activation.lower()

        if padding_strategy is not None:
            padding_strategy = padding_strategy.lower()

        assert activation in ["relu", "swish"] or activation is None

        self.padding_strategy = padding_strategy
        self.activation = activation
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Filter expansion (pointwise convolution)
        num_channels_expanded = in_channels * expansion_factor
        if expansion_factor != 1:
            self.filter_expansion = Conv1d(
                in_channels=in_channels,
                out_channels=num_channels_expanded,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=bias,
                padding_mode=padding_mode,
            )
            self.bn0 = torch.nn.BatchNorm1d(
                num_channels_expanded, eps=1e-3, momentum=0.1
            )

        # depthwise convolution (i.e. each channel separately, groups = num. channels)
        if self.padding_strategy == "same":
            padding_val = (kernel_size - 1) // 2
        else:
            padding_val = 0
        self.depthwise_conv = Conv1d(
            in_channels=num_channels_expanded,
            out_channels=num_channels_expanded,
            kernel_size=kernel_size,
            padding=padding_val,
            groups=num_channels_expanded,
            stride=stride,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bn1 = torch.nn.BatchNorm1d(num_channels_expanded, eps=1e-3, momentum=0.1)

        # Squeeze and Excitation
        if self.se_ratio:
            num_squeezed_channels = max(1, int(num_channels_expanded * self.se_ratio))
            self.se_reduce = Conv1d(
                in_channels=num_channels_expanded,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            )
            self.se_expand = Conv1d(
                in_channels=num_squeezed_channels,
                out_channels=num_channels_expanded,
                kernel_size=1,
            )

        # Reduction to final size (pointwise convolution)
        self.final_reduction = Conv1d(
            num_channels_expanded,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

        # Activation function:
        if activation == "swish":
            self.activation_f = torch.nn.SiLU()
        elif activation == "relu":
            self.activation_f = torch.nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Forward function of the MBConv1d block
        x = input

        # Filter expansion  (pointwise convolution)
        if self.expansion_factor != 1:
            x = self.filter_expansion(x)
            x = self.bn0(x)
            x = self.activation_f(x)

        # depthwise convolution (i.e. each channel separately, groups = num. channels)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation_f(x)

        # Squeeze and Excitation
        if self.se_ratio:
            x_squeezed = torch.nn.functional.adaptive_avg_pool1d(x, 1)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.activation_f(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Final reduction (pointwise convolution)
        x = self.final_reduction(x)
        x = self.bn2(x)

        # Skip connection:
        if (self.in_channels == self.out_channels) and (self.stride == 1):
            x = x + input

        return x

    @property
    def out_ch(self) -> int:
        return self.final_reduction.out_channels


#
# builder modules
#


def build_activation_module(activation_name: str) -> torch.nn.Module:

    # for cases where we don't want any activations
    if activation_name is None:
        return None

    if activation_name == "relu":
        return torch.nn.ReLU()

    if activation_name == "relu6":
        return torch.nn.ReLU6()

    if activation_name == "leaky_relu":
        return torch.nn.LeakyReLU()

    if activation_name == "gelu":
        return torch.nn.GELU()

    if activation_name == "silu":
        return torch.nn.SiLU()

    if activation_name == "elu":
        return torch.nn.ELU()

    if activation_name == "swish":
        return torch.nn.SiLU()

    if activation_name == "softplus":
        return torch.nn.Softplus()

    if activation_name == "sigmoid":
        return torch.nn.Sigmoid()

    raise ValueError(f"Activation name '{activation_name}' not found.")
