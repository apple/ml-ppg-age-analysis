#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Callable

from torch.utils.data import DataLoader, Dataset
import torch
from training.utils_main import seed_worker
from training.models.efficientnet import EfficientNet
from training.models.ssl_wrapper import (
    MomentumSslMultiViewModelWrapper,
)
from training.objectives.info_nce_reg import (
    InfoNceRegLoss,
)
from training.datasets.minimal_dataset import MinimalDataset

#
# build model
#


def build_model(model_name: str, model_params: dict) -> torch.nn.Module:
    """Pytorch model builder."""

    if model_name == "momentumssl_efficientnet":
        backbone_params = model_params.get("backbone_params", {})
        backbone = EfficientNet(**backbone_params)

        wrapper_params = model_params.get("wrapper_params", {})
        return MomentumSslMultiViewModelWrapper(backbone, **wrapper_params)

    raise ValueError(f"Model name '{model_name}' not found.")


#
# build loss
#


def build_loss(
    loss_name: str,
    loss_params: dict,
) -> torch.nn.Module:
    """Pytorch loss function builder."""

    if loss_name == "nce_reg":
        return InfoNceRegLoss(**loss_params)

    raise ValueError(f"Criterion name '{loss_name}' not found.")


#
# build datasets
#
def build_datasets(size_train=10000, size_val=100, size_test=100, dataset_params=None):
    """Pytorch dataet builder, register your datasets here"""
    if dataset_params is None:
        dataset_params = {}
    train_dataset = MinimalDataset(size=size_train, **dataset_params)
    val_dataset = MinimalDataset(size=size_val, **dataset_params)
    test_dataset = MinimalDataset(size=size_test, **dataset_params)
    return train_dataset, val_dataset, test_dataset


#
# build dataloaders
#


def build_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    batch_size_eval: int = 64,
    seed: int = 42,
    num_workers: int = 8,
    num_workers_eval: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    multiprocessing_context: str = "spawn",
    persistent_workers: bool = True,
    prefetch_factor: int = 16,
    prefetch_factor_eval: int = 16,
    collate_fn: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds training, validation, and test data loaders.

    Args:
        train_dataset (torch.utils.data.Dataset): Train torch dataset.
        val_dataset (torch.utils.data.Dataset): Val torch dataset.
        test_dataset (torch.utils.data.Dataset): Test torch dataset.
        batch_size (int, optional): The batch size for training.
        batch_size_eval (int, optional): The batch size for validation and testing.
        seed (int, optional): The random seed for data shuffling (if used for distributed samplers).
        num_workers (int, optional): The number of worker threads for data loading.
        num_workers_eval (int, optional): The number of worker threads for
            data loading during evaluation.
        pin_memory (bool): Whether to pin memory for faster data transfer.
        drop_last (bool optional): Whether to drop the last incomplete batch
            if the dataset size is not evenly divisible by the batch size.
        multiprocessing_context (str, optional): The multiprocessing context to use.
        persistent_workers (bool, optional): Whether to use persistent workers for data loading.
        prefetch_factor (int, optional): The number of prefetched batches to keep in memory.
        prefetch_factor_eval (int, optional): The number of prefetched batches to keep in memory during evaluation. D
        collate_fn (Callable, optional): A custom collate function to apply to batches of data.
    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """
    #
    # Training loader
    #

    train_loader = DataLoader(
        train_dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    #
    # Validation loader
    #

    val_loader = DataLoader(
        val_dataset,
        sampler=None,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval,
        pin_memory=pin_memory,
        drop_last=drop_last,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor_eval,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    #
    # Test loader
    #

    test_loader = DataLoader(
        test_dataset,
        sampler=None,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor_eval,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


#
# build learning rate scheduler
#


def build_learning_scheduler(optimizer, scheduler_name: str, scheduler_params: dict):
    """Pytorch learning rate scheduler builder."""

    if scheduler_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    raise ValueError(f"Learning schedule name '{scheduler_name}' not found.")


def build_optimizer(
    params, optimizer_name, optimizer_params: dict
) -> torch.optim.Optimizer:
    """Pytorch optimizer builder.
    params: iterator for params, most often this is model.parameters()
    optimizer_name (str): optimizer name
    optimizer_params (dict): optimizer parameters
    """

    if optimizer_name == "adam":
        return torch.optim.Adam(params, **optimizer_params)

    raise ValueError(f"Optimizer name '{optimizer_name}' not found.")
