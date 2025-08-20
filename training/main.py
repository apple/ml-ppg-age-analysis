#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse
import logging

import torch
import torchmetrics
import numpy as np
import pandas as pd
from typing import Tuple
import os

from training.builders import (
    build_datasets,
    build_loaders,
    build_loss,
    build_learning_scheduler,
    build_model,
    build_optimizer,
)
from training.utils_main import (
    compute_loss_and_output,
    export_torch_model,
    repeat_dataloader,
    clip_grad,
    get_momentum_value,
    to_gpu,
    import_config,
    set_deterministic_execution,
    set_random_seeds,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

#
# Training Entrypoint
#


def main(config):
    """Training loop entrypoint."""

    #
    # Initialize paths and filenames
    #
    (
        artifacts_dir,
        checkpoint_dir,
        checkpoint_filename,
        export_best_model_filename,
        export_final_model_filename,
    ) = init_dirs_and_filenames()

    #
    # Initialize or restore training state
    #
    logger.info("Initializing training state.")
    model, optimizer, lr_scheduler = init_training_state(
        config.model_name,
        config.model_params,
        config.optimizer_name,
        config.optimizer_params,
        config.scheduler_name,
        config.scheduler_params,
    )

    #
    # Initialize dataset
    #
    logger.info("Initializing datasets and data loaders.")
    training_dataset, validation_dataset, test_dataset = build_datasets(
        dataset_params=config.dataset_params
    )
    training_data, validation_data, test_data = build_loaders(
        training_dataset, validation_dataset, test_dataset, **config.dataloader_params
    )

    # For training we want infinite training data
    infinite_training_data = repeat_dataloader(training_data)

    #
    # Initialize training
    #
    criterion = build_loss(config.loss_name, config.loss_params)
    criterion = to_gpu(criterion)

    #
    # set up trainingmetrics
    #
    train_loss = to_gpu(torchmetrics.MeanMetric())

    #
    # Training loop
    #

    logger.info("Starting training loop. Training for %d batches.", config.total_steps)

    for step in range(config.total_steps):
        # Run one step of gradient descent
        loss, output, target = run_train_step(
            model,
            criterion,
            infinite_training_data,
            optimizer,
            step,
            config.total_steps,
            config.clip_grad_params,
            config.train_type_name,
            config.train_type_params,
        )
        train_loss.update(loss)

        if step > 0 and step % config.log_interval == 0:
            # log training metrics
            train_loss_value = train_loss.compute().item()
            logger.info(
                f"Training batches: {step}/{config.total_steps}. "
                f"Train loss: {train_loss_value:.4f}."
            )
            train_loss.reset()

        if step > 0 and step % config.validation_interval == 0:
            # Evaluate on validation data and log validation metrics
            logger.info("Starting evaluation at step %d.", step)
            val_loss_value = run_evaluation_loop(
                model,
                criterion,
                validation_data,
                prefix="val/",
            )
            logger.info(f"Val loss: {val_loss_value:.4f}.")

            # one step of lr scheduler every validation interval
            if lr_scheduler is not None:
                lr_scheduler.step()

            # export the model every validation interval
            export_best_model_filename_this = (
                f"{export_best_model_filename}_step{step}.pt"
            )
            logger.info(
                "Exporting new best model to '%s'.", export_best_model_filename_this
            )
            export_torch_model(export_best_model_filename_this, model)

            logger.info("Finished evaluation loop.")

    logger.info("Finished training.")
    logger.info("Exporting final model to '%s'.", f"{export_final_model_filename}.pt")
    export_torch_model(f"{export_final_model_filename}.pt", model)

    #
    # Test set evaluation
    #
    logger.info("Starting final evaluation on test set.")
    logger.info("Starting evaluation at step %d.", step)
    test_loss_value = run_evaluation_loop(
        model,
        criterion,
        test_data,
        prefix="test/",
    )
    logger.info(f"test loss: {test_loss_value:.4f}.")

    #
    # Store inferences on test set
    #
    logger.info("Running + storing inferences on test set.")
    dataset_params_inference = config.dataset_params
    dataset_params_inference.pop("augmentation_name")  # turn off augmentations
    _, _, test_dataset = build_datasets(dataset_params=dataset_params_inference)
    _, _, test_data = build_loaders(
        training_dataset, validation_dataset, test_dataset, **config.dataloader_params
    )
    # run inference on test data without augmentations
    test_inferences_df = run_evaluation_loop(
        model,
        criterion,
        test_data,
        return_inferences_df=True,
        prefix="inf/",
    )
    test_inferences_df.to_csv(os.path.join(artifacts_dir, "test_inferences.csv"))

    return


#
# training loop
#


def run_train_step(
    model,
    criterion,
    training_data,
    optimizer,
    step,
    total_steps,
    clip_grad_params,
    train_type_name,
    train_type_params,
):
    """Run a single training step on 1 batch of data."""
    model.train()
    data, target, _sample_ids = next(training_data)
    data, target = to_gpu(data), to_gpu(target)

    if train_type_name == "momentum":
        momentum_params = train_type_params["momentum_params"]
        # get momentum_value
        momentum_value = get_momentum_value(step, total_steps, **momentum_params)
        # momentum update
        model.momentum_update(momentum_value)

    # gradient update
    optimizer.zero_grad()
    loss, output = compute_loss_and_output(model, criterion, data, target)
    loss.backward()

    if clip_grad_params is not None:
        clip_grad(model, clip_grad_params)

    optimizer.step()

    return loss, output, target


#
# evaluation loop
#


@torch.no_grad()
def run_evaluation_loop(
    model,
    criterion,
    validation_data,
    return_inferences_df=False,
    prefix="val/",
):
    """Run inference with the provided model on
    the full validation/test dataset."""

    val_loss = to_gpu(torchmetrics.MeanMetric())

    outputs = []

    model.eval()
    for data, target, sample_id in validation_data:
        data, target = to_gpu(data), to_gpu(target)
        loss, output = compute_loss_and_output(model, criterion, data, target)
        val_loss.update(loss)

        if return_inferences_df:
            outputs += [output.detach().cpu().numpy()]

    val_loss_value = val_loss.compute().item()

    if return_inferences_df:
        # we keep the first view of the output
        # other views are just positive pairs for contrastive learning
        outputs = np.vstack(outputs)[:, :, 0]
        return pd.DataFrame(dict(embeddings=[o for o in outputs]))

    return val_loss_value


#
# Helpers
#


def init_dirs_and_filenames():
    """Create the necessary directories/filenames and return them"""

    artifacts_dir = "./artifacts"
    checkpoint_dir = "./checkpoints"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = os.path.join(checkpoint_dir, "checkpoint")
    export_best_model_filename = os.path.join(artifacts_dir, "best_model")
    export_final_model_filename = os.path.join(artifacts_dir, "final_model")

    return (
        artifacts_dir,
        checkpoint_dir,
        checkpoint_filename,
        export_best_model_filename,
        export_final_model_filename,
    )


def init_training_state(
    model_name,
    model_params,
    optimizer_name,
    optimizer_params,
    scheduler_name,
    scheduler_params,
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    """Initialize pytorch training state."""

    # build model
    model = build_model(model_name, model_params)

    # send the model to gpu
    model = to_gpu(model)

    # initialize optimizer
    optimizer = build_optimizer(model.parameters(), optimizer_name, optimizer_params)

    # initialize scheduler
    lr_scheduler = build_learning_scheduler(optimizer, scheduler_name, scheduler_params)

    return model, optimizer, lr_scheduler


def init_system(seed=42, deterministic_execution=False):
    set_random_seeds(seed)
    set_deterministic_execution(deterministic_execution)


#
# Run main
#


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("-c", "--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    # Get args
    args = parse_args()

    # Import config
    config = import_config(args.config)

    # Overwrite seed
    if args.seed:
        config.seed = args.seed

    # Init seeds/execution
    init_system(config.seed, config.deterministic_execution)

    # Start training
    main(config)
