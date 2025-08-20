#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from training.objectives.koleoreg import KoleoReg


class InfoNceRegLoss(nn.Module):
    """InfoNCE loss in addition to various regularizations
    - to add vicreg regularizations:
        use non-zero mu_std and nu_cov w/ gamma_koleo=0
    - to add koleo regularization:
        use non-zero gamma koleo w/ mu_std=0 and nu_cov=0
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_sim: float = 25.0,
        gamma_koleo: float = 0.0,
        mu_std: float = 25.0,
        nu_cov: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()

        self.temperature = temperature
        self.lambda_sim = lambda_sim
        self.gamma_koleo = gamma_koleo
        self.mu_std = mu_std
        self.nu_cov = nu_cov
        self.eps = eps

    def forward(
        self,
        latent_embeddings1: torch.Tensor,
        latent_embeddings2: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        lambda_sim: Optional[float] = None,
        gamma_koleo: Optional[float] = None,
        mu_std: Optional[float] = None,
        nu_cov: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        """InfoNCE loss in addition to various regularizations"""
        # set temperature at each evaluation -- if none use default
        temperature = self.temperature if temperature is None else temperature
        lambda_sim = self.lambda_sim if lambda_sim is None else lambda_sim
        gamma_koleo = self.gamma_koleo if gamma_koleo is None else gamma_koleo
        mu_std = self.mu_std if mu_std is None else mu_std
        nu_cov = self.nu_cov if nu_cov is None else nu_cov
        eps = self.eps if eps is None else eps
        return info_nce_reg_loss(
            latent_embeddings1,
            latent_embeddings2,
            labels=pids,
            temperature=temperature,
            lambda_sim=lambda_sim,
            gamma_koleo=gamma_koleo,
            mu_std=mu_std,
            nu_cov=nu_cov,
            eps=eps,
        )


def info_nce_reg_loss(
    latent_embeddings1: torch.Tensor,
    latent_embeddings2: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: Optional[float] = 0.1,
    lambda_sim: Optional[float] = 25.0,
    gamma_koleo: Optional[float] = 0.0,
    mu_std: Optional[float] = 25.0,
    nu_cov: Optional[float] = 1.0,
    eps: Optional[float] = 1e-4,
):
    """InfoNCE with various regularizations loss"""

    B, H = latent_embeddings1.shape

    # accumulate loss
    loss_acculumator = 0

    # the similarity loss
    contrast_loss = contrastive_loss(
        latent_embeddings1, latent_embeddings2, labels, temperature
    )
    # add to loss
    loss_acculumator += lambda_sim * contrast_loss

    # add koleo loss
    if gamma_koleo > 0.0:
        koleo_reg = KoleoReg()
        koleo_loss = 0.5 * (
            koleo_reg(latent_embeddings1) + koleo_reg(latent_embeddings2)
        )
        # add to loss
        loss_acculumator += gamma_koleo * koleo_loss

    if mu_std > 0.0 or nu_cov > 0.0:
        # zero-mean the views in the batch dim
        view1_array = latent_embeddings1 - latent_embeddings1.mean(0)  # (BxH)
        view2_array = latent_embeddings2 - latent_embeddings2.mean(0)  # (BxH)

        # variance regularization
        if mu_std > 0.0:
            view1_std = torch.sqrt(view1_array.var(dim=0) + eps)  # (H,)
            view2_std = torch.sqrt(view2_array.var(dim=0) + eps)  # (H,)
            std_loss = 0.5 * (
                torch.mean(F.relu(1 - view1_std)) + torch.mean(F.relu(1 - view2_std))
            )
            # add to loss
            loss_acculumator += mu_std * std_loss

        # covariance regularization
        if nu_cov > 0.0:
            # VicReg paper divides by B - 1, Barlow Twins divdes by B
            view1_cov = torch.mm(view1_array.T, view1_array) / (B - 1)  # (HxH)
            view2_cov = torch.mm(view2_array.T, view2_array) / (B - 1)  # (HxH)
            # this should be divided by (H**2 - H) but the original paper divides by H
            # to check
            cov_loss = (
                view1_cov[~torch.eye(H, dtype=bool)].pow(2).sum() / H
                + view2_cov[~torch.eye(H, dtype=bool)].pow(2).sum() / H
            )
            # add to loss
            loss_acculumator += nu_cov * cov_loss

    return loss_acculumator


def contrastive_loss(
    latent_embeddings1: torch.Tensor,
    latent_embeddings2: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    lambda_from_1_to_2: float = 0.5,
) -> torch.Tensor:
    """Calculate NCE Loss For Latent Embeddings in Batch
    Args:
        latent_embeddings1 (torch.Tensor): (embedding #1 from model (BxH) )
        latent_embeddings2 (torch.Tensor): (embedding #2 from model (BxH) )
        labels (torch.Tensor): labels of embeddings, embeddings with similar labels
            will be excluded from the negative samples, e.g., this can be subject ids
        temperature (float): hyperparametr for NCE loss
        lambda_from_1_to_2 (float): define the weights for the weighted sum
        of the loss from view 1 to 2 (view 1 is anchor/ground-truth) and view 2 to 1 (view 2 is anchor/ground-truth)
        this should be almost always set as 0.5, unless for advanced purposes
    Outputs:
        loss (torch.Tensor): scalar NCE loss
    """
    # if labels are not available, we assume they are all from different samples
    if labels is None:
        labels = torch.arange(
            latent_embeddings1.shape[0], device=latent_embeddings1.device
        )  # (B)

    # create label masking -> element (i, j) is False if it corresponds to the same label
    labels_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).to(
        latent_embeddings1.dtype
    )  # (BxB)

    # get input data from the two different views and normalize
    view1_array = F.normalize(latent_embeddings1, dim=1)  # (BxH)
    view2_array = F.normalize(latent_embeddings2, dim=1)  # (BxH)

    # SimCLR loss
    # dot products across views, (i, j) -> dot product of elemnt i in view 1 and element j in view 2
    sim_matrix = view1_array @ view2_array.transpose(0, 1)  # (BxB)
    # numerator for SimCLR loss: cosine sim, divided by temperature
    sim_matrix_exp = torch.exp(sim_matrix / temperature)  # (BxB)

    # dot products within view 1, (i, j) -> dot product of elemnt i in view 1 and element j in view 1
    self_sim_matrix1 = view1_array @ view1_array.transpose(0, 1)
    self_sim_matrix_exp1 = torch.exp(self_sim_matrix1 / temperature)
    # exclude elements that corresponds to the same instance (diagonals + collisions)
    self_sim_matrix_exp_masked1 = self_sim_matrix_exp1 * labels_mask

    # dot products within view 2, (i, j) -> dot product of elemnt i in view 2 and element j in view 2
    self_sim_matrix2 = view2_array @ view2_array.transpose(0, 1)
    self_sim_matrix_exp2 = torch.exp(self_sim_matrix2 / temperature)
    # exclude elements that corresponds to the same instance (diagonals + collisions)
    self_sim_matrix_exp_masked2 = self_sim_matrix_exp2 * labels_mask

    # denominator_loss1: for each i in view 1, sum up
    # similarities with other 2n - 1 samples (n from first term, and n-1 from the second term)
    denominator_loss1 = torch.sum(sim_matrix_exp, 1) + torch.sum(
        self_sim_matrix_exp_masked1, 1
    )
    # denominator_loss1: for each i in view2, sum up
    # similarities with other 2n - 1 samples (n from first term, and n-1 from the second term)
    denominator_loss2 = torch.sum(sim_matrix_exp, 0) + torch.sum(
        self_sim_matrix_exp_masked2, 0
    )

    # diagonals of cross-similarity == positive pairs
    diagonals = torch.diag(sim_matrix_exp)
    # put terms together for full loss
    loss_term1 = -torch.mean(torch.log(diagonals / denominator_loss1))  # (B) -> scalar
    loss_term2 = -torch.mean(torch.log(diagonals / denominator_loss2))  # (B) -> scalar
    # weighted average of loss from view 1 to 2, and from view 2 to 1
    # the standard is to be 0.5, unless for specific cases
    loss = lambda_from_1_to_2 * loss_term1 + (1 - lambda_from_1_to_2) * loss_term2

    return loss
