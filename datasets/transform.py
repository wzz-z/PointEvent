import copy
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from utils.registry import Registry


TRANSFORMS = Registry("transforms")


class Compose:
    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            return
        for transform in transforms:
            if transform is None:
                continue
            if isinstance(transform, dict):
                self.transforms.append(TRANSFORMS.build(transform))
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"Unsupported transform type: {type(transform)}")

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class Collect:
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.offset_keys = offset_keys_dict or dict(offset="coord")
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = {key: data_dict[key] for key in self.keys}
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy:
    def __init__(self, keys_dict=None):
        self.keys_dict = keys_dict or dict(coord="origin_coord", segment="origin_segment")

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            item = data_dict[key]
            if isinstance(item, np.ndarray):
                data_dict[value] = item.copy()
            elif isinstance(item, torch.Tensor):
                data_dict[value] = item.clone().detach()
            else:
                data_dict[value] = copy.deepcopy(item)
        return data_dict


@TRANSFORMS.register_module()
class ToTensor:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, str):
            return data
        if isinstance(data, bool):
            return torch.tensor(data)
        if isinstance(data, int):
            return torch.LongTensor([data])
        if isinstance(data, float):
            return torch.FloatTensor([data])
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            if data.dtype == np.uint32:
                data = data.astype(np.int64)
            return torch.from_numpy(data).long()
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        if isinstance(data, Mapping):
            return {key: self(value) for key, value in data.items()}
        if isinstance(data, Sequence):
            return [self(value) for value in data]
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Add:
    def __call__(self, data_dict):
        return data_dict


class GridSampleBase:
    def __init__(
        self,
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        keys=("coord", "strength", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = self._as_3d_grid_size(grid_size)
        self.hash_type = hash_type
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        if self.hash_type not in {"fnv", "ravel"}:
            raise ValueError(f"Unsupported hash_type: {hash_type}")

    @staticmethod
    def _as_3d_grid_size(grid_size):
        grid_size = np.asarray(grid_size, dtype=np.float32)
        if grid_size.ndim == 0:
            grid_size = np.repeat(grid_size, 3)
        if grid_size.shape[0] != 3:
            raise ValueError(f"grid_size should contain 3 values, got {grid_size}")
        return grid_size

    def hash(self, array):
        array = np.ascontiguousarray(array.astype(np.int64, copy=False))
        if self.hash_type == "fnv":
            return self.fnv_hash_vec(array)
        return self.ravel_hash_vec(array)

    @staticmethod
    def ravel_hash_vec(array):
        array = array.copy()
        array -= array.min(0)
        array = array.astype(np.uint64, copy=False)
        max_per_axis = array.max(0).astype(np.uint64) + 1
        keys = np.zeros(array.shape[0], dtype=np.uint64)
        for axis in range(array.shape[1] - 1):
            keys += array[:, axis]
            keys *= max_per_axis[axis + 1]
        keys += array[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(array):
        array = array.astype(np.uint64, copy=False)
        keys = np.full(array.shape[0], 14695981039346656037, dtype=np.uint64)
        for axis in range(array.shape[1]):
            keys *= np.uint64(1099511628211)
            keys ^= array[:, axis]
        return keys


@TRANSFORMS.register_module()
class GridSampleEventNorm(GridSampleBase):
    """Voxel sampling for normalized event coordinates.

    x, y, and t are divided by ``grid_size``. Polarity is included in the hash
    so events with different polarity are not merged into the same voxel.
    """

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            raise KeyError("GridSampleEventNorm requires 'coord'.")
        if "strength" not in data_dict:
            raise KeyError("GridSampleEventNorm requires 'strength'.")

        scaled_coord = data_dict["coord"] / self.grid_size
        grid_coord = np.floor(scaled_coord).astype(np.int64)
        raw_min_coord = grid_coord.min(0)
        grid_coord -= raw_min_coord
        scaled_coord -= raw_min_coord
        min_coord = raw_min_coord * self.grid_size

        polarity_key = np.rint(data_dict["strength"]).astype(np.int64) + 1
        hash_input = np.concatenate((grid_coord, polarity_key), axis=1)
        key = self.hash(hash_input)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(
            key_sort, return_inverse=True, return_counts=True
        )

        if self.mode == "train":
            data_dict = self._sample_train(
                data_dict, idx_sort, inverse, count, grid_coord, scaled_coord, min_coord
            )
            return data_dict
        if self.mode == "test":
            return self._sample_test(
                data_dict, idx_sort, inverse, count, grid_coord, scaled_coord, min_coord
            )
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _sample_train(
        self, data_dict, idx_sort, inverse, count, grid_coord, scaled_coord, min_coord
    ):
        start = np.cumsum(np.insert(count, 0, 0)[:-1])
        offset = np.random.randint(0, count.max(), count.size)
        idx_unique = idx_sort[start + offset % count]

        if "sampled_index" in data_dict:
            idx_unique = np.unique(np.append(idx_unique, data_dict["sampled_index"]))
            mask = np.zeros_like(data_dict["segment"]).astype(bool)
            mask[data_dict["sampled_index"]] = True
            data_dict["sampled_index"] = np.where(mask[idx_unique])[0]

        self._write_optional_fields(
            data_dict, idx_unique, idx_sort, inverse, grid_coord, scaled_coord, min_coord
        )
        for key in self.keys:
            data_dict[key] = data_dict[key][idx_unique]
        return data_dict

    def _sample_test(
        self, data_dict, idx_sort, inverse, count, grid_coord, scaled_coord, min_coord
    ):
        data_part_list = []
        start = np.cumsum(np.insert(count, 0, 0)[:-1])
        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse

        for i in range(count.max()):
            idx_part = idx_sort[start + i % count]
            data_part = dict(index=idx_part)
            if self.return_grid_coord:
                data_part["grid_coord"] = grid_coord[idx_part]
            if self.return_min_coord:
                data_part["min_coord"] = min_coord.reshape(1, 3)
            if self.return_displacement:
                data_part["displacement"] = self._displacement(
                    data_dict, idx_part, scaled_coord, grid_coord
                )
            for key, value in data_dict.items():
                data_part[key] = value[idx_part] if key in self.keys else value
            data_part_list.append(data_part)
        return data_part_list

    def _write_optional_fields(
        self, data_dict, idx, idx_sort, inverse, grid_coord, scaled_coord, min_coord
    ):
        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[idx]
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape(1, 3)
        if self.return_displacement:
            data_dict["displacement"] = self._displacement(
                data_dict, idx, scaled_coord, grid_coord
            )

    def _displacement(self, data_dict, idx, scaled_coord, grid_coord):
        displacement = scaled_coord - grid_coord - 0.5
        if self.project_displacement:
            displacement = np.sum(
                displacement * data_dict["normal"], axis=-1, keepdims=True
            )
        return displacement[idx]


@TRANSFORMS.register_module()
class SphereCrop:
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        if mode not in {"random", "center", "all"}:
            raise ValueError(f"Unsupported crop mode: {mode}")
        self.point_max = point_max
        self.sample_rate = sample_rate
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )
        if self.mode == "all":
            return self._crop_all(data_dict, point_max)
        if data_dict["coord"].shape[0] <= point_max:
            return data_dict

        if self.mode == "random":
            center = data_dict["coord"][np.random.randint(data_dict["coord"].shape[0])]
        else:
            center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
        idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), axis=1))[
            :point_max
        ]
        return self._slice_dict(data_dict, idx_crop)

    def _crop_all(self, data_dict, point_max):
        if "index" not in data_dict:
            data_dict["index"] = np.arange(data_dict["coord"].shape[0])
        if data_dict["coord"].shape[0] <= point_max:
            return [data_dict]

        data_part_list = []
        priority = np.random.rand(data_dict["coord"].shape[0]) * 1e-3
        covered = np.array([], dtype=data_dict["index"].dtype)
        while covered.size < data_dict["index"].shape[0]:
            center_idx = np.argmin(priority)
            dist2 = np.sum(np.square(data_dict["coord"] - data_dict["coord"][center_idx]), axis=1)
            idx_crop = np.argsort(dist2)[:point_max]
            data_crop = self._slice_dict(data_dict, idx_crop)
            data_crop["weight"] = dist2[idx_crop]
            data_part_list.append(data_crop)

            delta = np.square(1 - data_crop["weight"] / np.max(data_crop["weight"]))
            priority[idx_crop] += delta
            covered = np.unique(np.concatenate((covered, data_crop["index"])))
        return data_part_list

    @staticmethod
    def _slice_dict(data_dict, idx):
        original_len = data_dict["coord"].shape[0]
        sliced = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.shape[0] == original_len:
                sliced[key] = value[idx]
            elif isinstance(value, torch.Tensor) and value.shape[0] == original_len:
                sliced[key] = value[idx]
            else:
                sliced[key] = value
        return sliced
