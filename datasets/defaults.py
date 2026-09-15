from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from utils.logger import get_root_logger
from .transform import Compose, TRANSFORMS


class DefaultMultiScansDataset(Dataset):
    def __init__(
        self,
        split="test",
        data_root="data",
        gather_num=1,
        transform=None,
        test_mode=True,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.gather_num = gather_num
        self.transform = Compose(transform)
        self.loop = 1 if test_mode else loop
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if self.test_mode:
            self.scale2kitti = (
                TRANSFORMS.build(self.test_cfg.scale2kitti)
                if self.test_cfg.scale2kitti is not None
                else None
            )
            self.test_voxelize = (
                TRANSFORMS.build(self.test_cfg.voxelize)
                if self.test_cfg.voxelize is not None
                else None
            )
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop is not None
                else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        get_root_logger().info(
            "Loaded {} x {} samples from {}.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        raise NotImplementedError

    def get_multi_data(self, idx):
        raise NotImplementedError

    def get_data_name(self, idx):
        raise NotImplementedError

    def prepare_test_data(self, idx):
        data_dict = self.get_multi_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=self.get_data_name(idx))

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = [aug(deepcopy(data_dict)) for aug in self.aug_transform]

        fragment_list = []
        for data in data_dict_list:
            if self.scale2kitti is not None:
                data = self.scale2kitti(data)
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]

            for data_part in data_part_list:
                if self.test_crop is not None:
                    fragment_list += self.test_crop(data_part)
                else:
                    fragment_list.append(data_part)

        result_dict["fragment_list"] = [
            self.post_transform(fragment) for fragment in fragment_list
        ]
        return result_dict

    def __getitem__(self, idx):
        if not self.test_mode:
            raise RuntimeError("This supplemental package only supports test_mode=True.")
        return self.prepare_test_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
