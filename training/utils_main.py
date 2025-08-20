#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
from typing import Any
import random
import numpy as np
import collections
from training import configs


def import_config(config_name: str):
    """Import config by name."""
    config_class = getattr(configs, config_name)
    return config_class()


def repeat_dataloader(dataloader):
    """
    Returns a generator that yields an indefinite sequence of batches from a
    :obj:`torch.utils.data.DataLoader` by iterating over the dataloader for an unlimited
    number of times.  If the dataloader also has a :attr:`sampler` with a
    :meth:`set_epoch` method, the generator calls
    ``dataloader.sampler.set_epoch(epoch)`` before each pass over the data.

    Args:
        dataloader (:obj:`torch.utils.data.DataLoader`): A dataloader for generating
            batches.

    Returns:
        A generator that generates an infinite sequence of batches.
    """
    assert len(dataloader) > 0, "dataloader cannot be empty"
    epoch = 0
    while True:
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1


def compute_loss_and_output(model, criterion, data, target):
    """compute loss and output for model"""

    # compute output
    output_student1, output_teacher1 = model(data)
    output_student2, output_teacher2 = model(torch.flip(data, dims=[-1]))

    # construct output
    output = torch.stack([output_student1, output_teacher1], dim=-1)

    # calculate loss
    loss = 0.5 * (
        criterion(output_student1, output_teacher1, target)
        + criterion(output_student2, output_teacher2, target)
    )

    return loss, output


#
# utils torch
#


def export_torch_model(filename: str, model: torch.nn.Module) -> None:
    """Export final model state to filename."""
    torch.save(model.state_dict(), filename)


def is_cuda_available() -> bool:
    """check if cuda is available"""
    return torch.cuda.is_available()


def set_random_seeds(seed: int = 42):
    """Set all the random seeds for reproducible training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed(seed)


def seed_worker(worker_id):
    """Set the seed for each worker in the dataloader for reproducibility:
    more info: https://pytorch.org/docs/stable/data.html#data-loading-randomness
    Under the hood, after initializing each worker, PyTorch updates the seed by
    base_seed + worker_id, so worker_seed here should automatically be different across workers,
    hence not using worker_id (but we need it as input),
    but we should manually set other libraries seed here
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_deterministic_execution(deterministic_execution: bool = False):
    """Ensures deterministic execution of PyTorch algorithms and cudnn operations,
    this should have deterministic_execution=False in most occasions unless for exact reproducibility experiments/debugging.
    https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    """
    if deterministic_execution:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # the following is only necessary for specific occasions
        # for deterministic linear algebra algs of cuBLAS
        # but requires setting a new env variable
        # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
        # torch.use_deterministic_algorithms(True)


def to_gpu(obj: Any):
    """Move obj to the appropriate GPU if available."""
    return nested_to_gpu(obj)


def nested_to_gpu(obj: Any):
    """Recursively move nested obj to the appropriate GPU if available."""

    # if gpu not available return
    # torch does not support python set objects
    if not is_cuda_available() or isinstance(obj, set) or obj is None:
        return obj

    # do a recursive run if on cuda
    if isinstance(obj, dict) or isinstance(obj, collections.abc.Mapping):
        return {k: nested_to_gpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [nested_to_gpu(v) for v in obj]

    return obj.cuda()


def clip_grad(model: torch.nn.Module, clip_grad_params):
    """clip gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), **clip_grad_params)


def disable_grad(model: torch.nn.Module):
    """disables gradient for a torch model"""

    for param in model.parameters():
        param.requires_grad = False


def enable_grad(model: torch.nn.Module):
    """
    enables gradient for a torch model,
    this can be used after disable_grad()
    to re-enable the gradient updates
    """

    for param in model.parameters():
        param.requires_grad = True


def update_momentum_step(
    model: torch.nn.Module, model_momentum: torch.nn.Module, m: float
):
    """
    updates model_momentum with model weights using the momentum update rule:
    model_momentum = m * model_momentum + (1 - m) * model
    """
    for param, param_momentum in zip(model.parameters(), model_momentum.parameters()):
        param_momentum.data = m * param_momentum.data + (1 - m) * param.data


def get_momentum_value(step, total_scale_invariant_steps, **kwargs):
    """
    returns the momentum value given a momentum schedule
    Inputs:
        - step (int): current step
        - total_scale_invariant_steps (int): total steps (scale invariant)
        - kwargs: dictionary for momentum parameters
        - momentum_value_base (float): base momentum value
        - momentum_schedule (str): momentum schedule (supported values: constant or cosine_warmup)
    Ouptuts:
        - momentum_value (float): momentum value given the schedule
    """

    # necessary keys in kwargs
    momentum_schedule = kwargs.get("momentum_schedule", "constant")
    momentum_value_base = kwargs.get("momentum_value_base", None)

    # constant momentum
    if momentum_schedule == "constant":
        return momentum_value_base

    # cosine warm up momentum
    if momentum_schedule == "cosine_warmup":
        """
        similar to byol paper:
        https://arxiv.org/abs/2006.07733
        """
        return (
            1
            - (1 - momentum_value_base)
            * (np.cos(np.pi * (step / total_scale_invariant_steps)) + 1)
            / 2
        )

    # freeze for `freeze_steps` and then constant
    if momentum_schedule == "freeze_then_constant":
        freeze_steps = kwargs.get("freeze_steps", None)
        if step <= freeze_steps:
            return 1.0
        else:
            return momentum_value_base

    raise NotImplementedError(
        f"momentum_schedule: {momentum_schedule} is not implemented."
    )
