#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn.functional as F
from torch import nn


class KoleoReg(nn.Module):
    """
    Kozachenko-Leonenko (KoLeo) regularization used in DINO-V2
    sources:
    https://arxiv.org/pdf/1806.03198.pdf
    https://arxiv.org/pdf/2304.07193.pdf

    Args:
        do_normalize (bool): normalize the embeddings or not
                             use False if normalized outside
        eps (float): eps for precision operations
    """

    def __init__(self, do_normalize: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.pdist = nn.PairwiseDistance(2, eps=self.eps)
        self.do_normalize = do_normalize

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        KoLeo regularization
        Args:
            embeddings (torch.Tensor): BxH
        Returns:
            torch.Tensor: scalar KoLeo loss/regularization.
        """
        B = embeddings.shape[0]  # batch size

        # l-2 normalize the embeddings
        if self.do_normalize:
            embeddings = F.normalize(embeddings, eps=self.eps, dim=1)

        # get the max inner dot product to find min distance
        dots = torch.mm(embeddings, embeddings.t())
        # trick to fill diagonal with -1
        dots.view(-1)[:: (B + 1)].fill_(-1)
        # get the min dist indices
        _, min_dist_idx = torch.max(dots, dim=1)

        # get the pair wise distance
        distances = self.pdist(embeddings, embeddings[min_dist_idx])  # BxD, BxD -> B
        loss = -torch.log(distances + self.eps).mean()

        return loss
