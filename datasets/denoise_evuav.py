import os
from pathlib import Path

import numpy as np

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset


@DATASETS.register_module()
class denoise_evuav(DefaultMultiScansDataset):
    """EV-UAV point-wise binary event detection dataset.

    Each npz file is expected to contain an ``evs_norm`` array with columns:
    x, y, t, polarity, label, name. Only the first five columns are used.
    """

    def __init__(
        self,
        split="train",
        data_root="data",
        gather_num=1,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        polarity_to_signed=False,
    ):
        self.ignore_index = ignore_index
        self.polarity_to_signed = polarity_to_signed
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            gather_num=gather_num,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        data_root = Path(self.data_root)
        splits = [self.split] if isinstance(self.split, str) else list(self.split)
        data_list = []
        for split in splits:
            split_dir = data_root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"EV-UAV split directory not found: {split_dir}")
            data_list.extend(sorted(str(path) for path in split_dir.rglob("*.npz")))
        if not data_list:
            raise RuntimeError(f"No npz files found for split {self.split} in {data_root}")
        return data_list

    def get_data_name(self, idx):
        path = Path(self.data_list[idx % len(self.data_list)])
        return f"{path.parent.name}+{path.stem}"

    def get_multi_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with np.load(data_path) as data:
            if "evs_norm" not in data:
                raise KeyError(f"{data_path} does not contain 'evs_norm'")
            evs_norm = data["evs_norm"]

        if evs_norm.ndim != 2 or evs_norm.shape[1] < 5:
            raise ValueError(
                f"{data_path} expected evs_norm shape (N, >=5), got {evs_norm.shape}"
            )

        coord = evs_norm[:, 0:3].astype(np.float32, copy=False)
        polarity = evs_norm[:, 3:4].astype(np.float32, copy=True)
        if self.polarity_to_signed:
            polarity = polarity * 2.0 - 1.0

        segment = evs_norm[:, 4].astype(np.int64, copy=False).reshape(-1)
        valid = (segment == 0) | (segment == 1) | (segment == self.ignore_index)
        if not np.all(valid):
            bad = np.unique(segment[~valid])[:10]
            raise ValueError(f"{data_path} contains labels outside 0/1: {bad}")

        return dict(coord=coord, strength=polarity, segment=segment)

    @staticmethod
    def get_learning_map_inv(ignore_index):
        return {
            ignore_index: ignore_index,
            0: 0,
            1: 1,
        }
