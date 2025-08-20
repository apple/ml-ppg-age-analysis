#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from training.models.utils import build_activation_module


class MLP(nn.Module):
    def __init__(
        self,
        mlp_layers_list: list,
        dropout_p: float = 0.25,
        do_batchnorm: bool = False,
        hidden_activation_name: str = "relu",
        final_activation_name: str = None,
        is_time_distributed=False,
    ):
        """
        mlp_layers_list: list of the layers in mlp with this format [input_dim, *hidden_layers, output_dim],
                         if only two elements, it behaves as a linear matrix mult from input_dim to output_dim
        dropout_p: probability of dropout for hidden layers
        do_batchnorm: whether to have batchnorm for hidden layers or not
        hidden_activation_name: name for hidden layers activations, None for no activation
        final_activation_name: name for final layer activations, None for no activation
        is_time_distributed: whether to use Conv1d layers (with kernel size 1) instead of Linear.
                             When True, input: (B, C, T), output: (B, C_out, T). This is useful for sequence outputs.
                             When False, input: (B, C), output: (B, C_out).
        """
        super(MLP, self).__init__()

        # build activations
        self.hidden_activation = build_activation_module(hidden_activation_name)
        self.final_activation = build_activation_module(final_activation_name)

        # get the number of layers and check
        n_layers = len(mlp_layers_list)
        self.mlp_layers_list = mlp_layers_list

        # checks
        if n_layers < 2:
            raise ValueError(
                "mlp_layers_list requires at least 2 elements -- input and output dimensions."
            )

        if self.hidden_activation is None and n_layers > 2:
            raise ValueError(
                "Nonlinear hidden activation for n_layers > 2 is required."
            )

        # mlp architecture
        mlp = []
        for i in range(n_layers - 1):
            if is_time_distributed:
                # MLP distributed temporally using 1d conv with kernel size 1
                mlp.append(
                    nn.Conv1d(
                        in_channels=self.mlp_layers_list[i],
                        out_channels=self.mlp_layers_list[i + 1],
                        kernel_size=1,
                    )
                )
            else:
                mlp.append(
                    nn.Linear(self.mlp_layers_list[i], self.mlp_layers_list[i + 1])
                )

            # for hidden layers
            if i != n_layers - 2:
                if do_batchnorm:
                    mlp.append(
                        nn.BatchNorm1d(
                            self.mlp_layers_list[i + 1], eps=1e-3, momentum=0.1
                        )
                    )
                mlp.append(self.hidden_activation)
                mlp.append(nn.Dropout(dropout_p))

        # for final layer
        if self.final_activation:
            mlp.append(self.final_activation)

        # build self.mlp
        self.mlp = nn.Sequential(*mlp)

        return

    def forward(self, h):

        z = self.mlp(h)

        return z
