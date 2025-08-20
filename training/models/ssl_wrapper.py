#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import copy
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from training.models.mlp import MLP
from training.utils_main import (
    disable_grad,
    update_momentum_step,
)


class SslMultiViewModelWrapper(nn.Module):
    """Self Supervised Multi View Wrapper Model.

    This model wraps any underlying backbone model to support
    multi-view ssl training by running the backbone model in
    addition to a mlp head on each view.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        mlp_head_layers: Optional[list] = None,
        dropout_p: Optional[float] = 0.25,
        do_batchnorm: Optional[bool] = False,
    ):
        """Constructor for SslWrapperModel.

        Args:
            backbone (torch.nn.Module): backbone model.
            mlp_head_layers (list, optional): mlp head's architecture,
                A list of integers determining the number of units in each layer of the
                the mlp head, format: [embedding_size, *hidden_size, representation_size]
            dropout_p (float, optional): dropout probability in the mlp head
            do_batchnorm (bool, optional): whether to add batchnorm in the mlp head
        """
        super().__init__()

        self.backbone = backbone
        # build the mlp head, if None it will be ignored in the forward pass
        self.mlp_head = (
            MLP(mlp_head_layers, dropout_p, do_batchnorm)
            if mlp_head_layers is not None
            else None
        )

    def forward(self, x: Union[List, Tuple, torch.Tensor]) -> torch.Tensor:
        """Run the backbone model and the mlp head on each augmented view.

        Args:
            x: Input, which is either:
                A list or tuple size num_views, where each element is the input, or
                A torch.Tensor of shape (batch_size, *model_input_sizes, num_views)
        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, representation_size, num_views)
        """
        # Move num_views to the first dimension for compatibility with other input types
        if isinstance(x, torch.Tensor):
            x = x.movedim(-1, 0)  # shape: num_views, batch_size, *model_input_sizes

        num_views = len(x)

        # We could do this in batch, but there is no guarantee that
        # (B * num_views) samples can fit into memory.
        # Instead we compute (B,) samples sequentially
        outputs = []
        for idx in range(num_views):
            # If x is torch.Tensor, x[idx] is x[idx, ...], which returns a view
            # data of shape (B, ...)
            # If x is a list, x[idx] returns the an element of the list, which is again
            # a view of data
            view_output = self.backbone(x[idx])  # shape: batch_size, embedding_size
            if self.mlp_head is not None:
                view_output = self.mlp_head(
                    view_output
                )  # shape: batch_size, representation_size
            outputs.append(view_output)

        return torch.stack(
            outputs, dim=-1
        )  # shape: batch_size, representation_size, num_views


class MomentumSslMultiViewModelWrapper(SslMultiViewModelWrapper):
    """Self Supervised Multi View Wrapper Model.

    This model wraps any underlying backbone model to support
    momentum ssl training by running the backbone model in
    addition to a mlp head on each view.

    The main difference with the parent class is that this model
    enables momentum training as well.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        mlp_head_layers: list,
        mlp_pred_layers: Optional[list] = None,
        dropout_p: Optional[float] = 0.0,
        do_batchnorm: Optional[bool] = False,
    ):
        """Constructor for SslWrapperModel.

        Args:
            backbone (torch.nn.Module): backbone model.
            mlp_head_layers (list): mlp head's architecture,
                A list of integers determining the number of units in each layer of the
                the mlp head, format: [embedding_size, *hidden_size, representation_size]
            mlp_pred_layers (list, optional): prediction mlp head's architecture,
                A list of integers determining the number of units in each layer of the
                the mlp head, format: [representation_size, *hidden_size, representation_size]
            dropout_p (float, optional): dropout probability in the mlp head
            do_batchnorm (bool, optional): whether to add batchnorm in the mlp head
        """
        super().__init__(backbone, mlp_head_layers, dropout_p, do_batchnorm)

        self.mlp_pred = (
            MLP(mlp_pred_layers, dropout_p, do_batchnorm)
            if mlp_pred_layers is not None
            else None
        )

        # create the backbone momentum copy
        self.backbone_momentum = copy.deepcopy(self.backbone)
        disable_grad(self.backbone_momentum)
        # create the mlp head momentum copy
        self.mlp_head_momentum = copy.deepcopy(self.mlp_head)
        disable_grad(self.mlp_head_momentum)

    def forward_student(self, view_student):
        """
        Performs student (online network) forward computation:
        view_student -> online backbone -> online mlp head -> online mlp pred -> output_student
        """
        output_student = self.backbone(view_student)
        output_student = self.mlp_head(output_student)
        if self.mlp_pred is not None:
            output_student = self.mlp_pred(output_student)

        return output_student

    def forward_teacher(self, view_teacher):
        """
        Performs teacher (momentum network) forward computation:
        view_teacher -> momentum backbone -> momentum mlp head -> output_teacher
        """
        with torch.no_grad():
            output_teacher = self.backbone_momentum(view_teacher)
            output_teacher = self.mlp_head_momentum(output_teacher)

        return output_teacher

    def forward(self, x: Union[List, Tuple, torch.Tensor]) -> torch.Tensor:
        """Run the backbone model and the mlp head on each augmented view.
           first view goes through the actual model, the rest of the views
           go through the momentum updated version of the actual model
        Args:
            x: Input, which is either:
                A list of size 2, where each element is the correct model input
                A torch.Tensor (batch_size, *model_input_sizes, 2) as (..., [view_student, view_teacher])
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors of shape
                (batch_size, representation_size)
        """
        # Move num_views to the first dimension for compatibility with other input types
        if isinstance(x, torch.Tensor):
            x = x.movedim(-1, 0)  # shape: 2, batch_size, *model_input_sizes
        if len(x) != 2:
            raise ValueError("Momentum training only supports two views of a sample")

        # If x is torch.Tensor, x[0] is x[0, ...], which returns the first view
        # data of shape (B, ...), similarly for x[1] which returns the second view
        # If x is a list, x[0] returns the first element of the list, which is again
        # the first view of data, and x[1] is the second view
        view_student = x[0]
        view_teacher = x[1]
        # get outputs
        output_student = self.forward_student(view_student)
        output_teacher = self.forward_teacher(view_teacher)

        return output_student, output_teacher.detach()

    def momentum_update(self, momentum_value):
        # momentum update for the backbone
        update_momentum_step(
            self.backbone,
            self.backbone_momentum,
            m=momentum_value,
        )
        # momentum update for the mlp head
        update_momentum_step(
            self.mlp_head,
            self.mlp_head_momentum,
            m=momentum_value,
        )
