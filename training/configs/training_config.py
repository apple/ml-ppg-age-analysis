#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional


class TrainingPpgSsl:

    dataset_params: dict = {
        "do_zscore": True,
        "augmentation_name": "stochastic_cascade",
        "augmentation_config": {
            "augmentation_prob_dict": {
                "cut_out": 0.4,
                "magnitude_warp": 0.25,
                "time_warp": 0.15,
                "add_gaussian_noise": 0.25,
                "channel_permute": 0.25,
            },
        },
    }

    # set to 256 for similar results to https://arxiv.org/pdf/2312.05409
    # here set to 16 for sped-up run on local device for fixed # of iters
    dataloader_params: dict = {"batch_size": 16}

    embedding_dim = 256
    input_shape = (4, 3840, 2)

    model_params: dict = {
        "backbone_params": {
            "input_shape": input_shape[0],
            "output_shape": embedding_dim,
            "kernel_size": 3,
            "activation": "swish",
            "expansion_factor": 7,
            "se_ratio": 0.25,
        },
        "wrapper_params": {
            "mlp_head_layers": [embedding_dim, 1024, 128],
            "dropout_p": 0.1,
        },
    }
    model_name: str = "momentumssl_efficientnet"

    loss_name: str = "nce_reg"
    loss_params: dict = {
        "temperature": 0.04,
        "lambda_sim": 1.0,
        "mu_std": 0.0,
        "nu_cov": 0.0,
        "gamma_koleo": 0.1,
    }

    train_type_name = "momentum"
    train_type_params: dict = {
        "momentum_params": {
            "momentum_value_base": 0.99,
            "momentum_schedule": "constant",
        }
    }

    # set to higher numbers depending on your compute budget
    # here set to small numbers for sped-up example run
    total_steps: int = 10  # realistic example: 10**7
    checkpoint_interval: int = 1  # realistic example: 500
    validation_interval: int = 2  # realistic example: 1000
    log_interval: int = 1  # realistic example: 500

    scheduler_name = "step_lr"
    scheduler_params = {"step_size": 125, "gamma": 0.5}
    optimizer_name: str = "adam"
    optimizer_params: dict = {
        "lr": 0.001,
        "weight_decay": 0.00001,
    }
    clip_grad_params: Optional[dict] = None

    seed: int = 42
    deterministic_execution: bool = False
