#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from torch.utils.data import Dataset
import numpy as np
from training.datasets.augmentations import TimeseriesAugmentations
from sklearn.preprocessing import StandardScaler


class MinimalDataset(Dataset):
    """
    minimal dataset for testing purposes
    creates random signals of shape (4, 3840)
    and random subject ids
    """

    def __init__(
        self,
        size=1000,
        do_zscore=True,
        augmentation_name: str = "identity",
        augmentation_config: dict = None,
    ):
        self.size = size
        self.do_zscore = do_zscore
        # augmentation params
        self.augmentation_tool = TimeseriesAugmentations()
        self.augmentation_name = augmentation_name
        self.augmentation_config = augmentation_config or {}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        signal_1, signal_2 = np.random.randn(4, 3840), np.random.randn(4, 3840)
        signal_1, signal_2 = self.transform(signal_1), self.transform(signal_2)
        pid = np.random.randint(1, 10000)  # random subject ids
        return np.stack([signal_1, signal_2], axis=-1).astype(np.float32), pid, idx

    def transform(self, segment):
        if self.do_zscore:
            segment = zscore_segment(segment)
        if self.augmentation_name != "identity":
            # create a view/augmentation of the current segment
            # the augmentation functions receive (time, num_channels),
            # hence the .T operators
            segment = getattr(self.augmentation_tool, self.augmentation_name)(
                segment.T, **self.augmentation_config
            ).T

        return segment


#
# utility functions
#


def zscore_segment(x: np.ndarray) -> np.ndarray:
    """Z-score a segment, per-channel basis"""
    zscore_tool = StandardScaler()
    x = zscore_tool.fit_transform(x.T).T

    return x
