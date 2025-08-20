#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


class TimeseriesAugmentations(object):
    def __init__(self):
        """
        empty return
        """
        return

    def identity(self, x):
        return x

    def cut_out(self, x, proportion_range=[0.4, 0.6]):
        """
        This method cuts out some part of the data and pad with zeros
        x: (np.array), (time, dim), the original signal
        proportion_range: (list), (2, ), range for how much of the time axis to be padded with zeros,
        e.g., [0.3, 0.5] will result in a cut out signal where 30-50% (random length, random start point)
        of it is padded with zeros
        """
        n_steps = x.shape[0]
        prop_low = proportion_range[0]
        prop_high = proportion_range[1]
        # pick how long we want to cut out
        cut_random_length = np.random.randint(
            int(prop_low * n_steps), int(prop_high * n_steps)
        )
        # pick the random start cut out time
        cut_random_start = np.random.randint(0, n_steps - cut_random_length)
        # augment the signal
        x_augmented = x.copy()
        x_augmented[cut_random_start : (cut_random_start + cut_random_length), :] = (
            np.zeros((cut_random_length, x.shape[1]))
        )

        return x_augmented

    def add_gaussian_noise(self, x, sigma_range=[0.05, 0.1]):
        """
        This method adds Gaussian noise to the signal
        x: (np.array), (time, dim), the original signal
        sigma_range: list, (2, ), range for std (sigma) of Gaussian noise,
        e.g., [0.1, 0.2] adds Gaussian noise whose std is between 0.1 and 0.2
        """
        sigma_low = sigma_range[0]
        sigma_high = sigma_range[1]
        # pick the random noise value
        sigma_value = np.random.uniform(sigma_low, sigma_high)
        # generate the random noise
        noise_to_add = sigma_value * np.random.randn(x.shape[0], x.shape[1])
        # augment the signal
        x_augmented = x + noise_to_add

        return x_augmented

    def magnitude_warp(self, x, scale_sigma=0.2, num_knots=5):
        """
        This method performs magnitude warping
        Advanced version of magnitude_scale()
        x: (np.array), (time, dim), the original signal
        scale_sigma: (float), the standard deviation of scaling at each knot
        -- for each knot a scaling factor will be picked from a Gaussian N ~ (1, sigma)
        num_knots: (int), number of knots to perform magnitude warping for
        """
        scale_knot = np.random.normal(
            loc=1, scale=scale_sigma, size=(num_knots + 2, 1)
        )  # before interpolation
        scale_knot_interp = multi_dim_interp(scale_knot, x.shape[0], kind="cubic")
        x_augmented = x * scale_knot_interp
        return x_augmented

    def time_warp(self, x, scale_sigma=0.1, num_knots=5):
        """
        This method performs time warping
        Advanced version of time_scale()
        x: (np.array), (time, dim), the original signal
        scale_sigma: (float)), the standard deviation of scaling at each knot
        -- for each knot a scaling factor will be picked from a Gaussian N ~ (1, sigma)
        num_knots: int, number of knots to perform time warping for
        """
        orig_steps = np.arange(x.shape[0])

        # pick a random warp location in time for each knot
        random_warps = np.random.normal(
            loc=1.0, scale=scale_sigma, size=(num_knots + 2, x.shape[1])
        )
        # pick the warp step locations based on teh number of knots
        warp_steps = (
            np.ones((x.shape[1], 1))
            * (np.linspace(0, x.shape[0] - 1.0, num=num_knots + 2))
        ).T
        # start warping for each dimension
        x_augmented = np.zeros_like(x)
        for dim in range(x.shape[1]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps[:, dim]
            )(orig_steps)
            # pick the scale based on the length of the signal
            scale = (x.shape[0] - 1) / time_warp[-1]
            x_augmented[:, dim] = np.interp(
                orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1), x[:, dim]
            ).T
        return x_augmented

    def channel_permute(self, x):
        """
        This method permutes the channels in the original signal
        x: (np.array), (time, dim), the original signal
        """
        n_channels = x.shape[1]
        x_augmented = x[:, np.random.permutation(n_channels)]

        return x_augmented

    # cascade augmentations
    def stochastic_cascade(
        self, x, augmentation_prob_dict=None, augmentation_config=None
    ):
        """
        This method performs a stochastic cascade of basic augmentations on a given signal
        x: (np.array), (time, dim), the original signal
        augmentation_prob_dict: (dict), keys are augmentation, values are probability of
        the augmentation being applied in each run
        augmentations: (list), list of augmentations
        augs_config: (dict), a dictionary containing configs for the augmentations
        """
        if augmentation_prob_dict is None:
            augmentation_prob_dict = {
                "cut_out": 0.8,
                "amplitude_scale": 0.7,
                "time_scale": 0.5,
                "add_gaussian_noise": 0.5,
            }
        if augmentation_config is None:
            augmentation_config = get_default_config_for_augmentations()

        x_augmented = x
        for aug, aug_prob in augmentation_prob_dict.items():
            # flip a coin to see whether to apply that augmentation
            random_flip = np.random.binomial(1, aug_prob, 1)[0]
            if random_flip:
                x_augmented = getattr(self, aug)(
                    x_augmented, **augmentation_config[aug]
                )

        return x_augmented


#
# default configs
#


def get_default_config_for_augmentations():
    augs_config = {
        "identity": {},
        "cut_out": {"proportion_range": [0.3, 0.5]},
        "add_gaussian_noise": {"sigma_range": [0.05, 0.1]},
        "magnitude_warp": {"scale_sigma": 0.2, "num_knots": 5},
        "time_warp": {"scale_sigma": 0.1, "num_knots": 5},
        "channel_permute": {},
        "stochastic_cascade": {
            "augmentation_prob_dict": {
                "cut_out": 0.8,
                "amplitude_scale": 0.7,
                "time_scale": 0.5,
                "add_gaussian_noise": 0.5,
            }
        },
    }
    return augs_config


#
# utility functions
#


def multi_dim_interp(x, interp_length, kind="cubic"):
    x_interp = np.empty(shape=(interp_length, x.shape[1]), dtype="float32")
    # create time
    t_this = np.arange(0, np.shape(x)[0])
    t_this_interp = np.linspace(0, t_this[-1], interp_length)
    for d in range(x.shape[1]):  # for each dimension
        f_interp = interp1d(t_this, x[:, d], kind=kind)
        x_d_interp = f_interp(t_this_interp)
        x_interp[:, d] = x_d_interp

    return x_interp
