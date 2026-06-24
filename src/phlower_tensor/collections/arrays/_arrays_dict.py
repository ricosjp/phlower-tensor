from collections import defaultdict

import numpy as np
import torch

from phlower_tensor._array import IPhlowerArray
from phlower_tensor._tensor import PhlowerTensor, phlower_tensor


class _PhlowerSequenceArray:
    def __init__(self, name: str, data: list[IPhlowerArray]) -> None:
        self._name = name
        self._data = data
        assert len(self._data) > 0

        self._is_sparse = self._reduce_flag("is_sparse")
        self._is_time_series = self._reduce_flag("is_time_series")
        self._is_voxel = self._reduce_flag("is_voxel")

    def _reduce_flag(self, attr_name: str) -> bool:
        flags = np.unique(np.array([getattr(v, attr_name) for v in self._data]))
        if len(flags) != 1:
            raise ValueError(f"{attr_name} is not unique in the {self._name}")
        return flags.item()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def is_time_series(self) -> bool:
        return self._is_time_series

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    def to_phlower_tensors(
        self,
        device: str | torch.device,
        non_blocking: bool,
        disable_dimensions: bool,
    ) -> list[PhlowerTensor]:
        return [
            phlower_tensor(
                v.to_tensor(
                    device=device,
                    non_blocking=non_blocking,
                ),
                dimension=None if disable_dimensions else v.dimension,
                is_time_series=v.is_time_series,
                is_voxel=v.is_voxel,
            )
            for v in self._data
        ]


class SequencedDictArray:
    def __init__(self, data: list[dict[str, IPhlowerArray]]) -> None:
        _sequece_array_dict: dict[str, list[IPhlowerArray]] = defaultdict(list)

        for dict_data in data:
            for k, v in dict_data.items():
                _sequece_array_dict[k].append(v)

        for k, vv in _sequece_array_dict.items():
            if len(vv) != len(data):
                raise ValueError(
                    f"the number of data is not consistent in {k}. "
                    f"Please check that {k} exists in all data"
                )

        self._phlower_sequece_dict = {
            k: _PhlowerSequenceArray(k, arr)
            for k, arr in _sequece_array_dict.items()
        }

    def get_names(self) -> list[str]:
        return list(self._phlower_sequece_dict.keys())

    def to_phlower_tensors_dict(
        self,
        device: str | torch.device,
        non_blocking: bool,
        disable_dimensions: bool = False,
    ) -> dict[str, list[PhlowerTensor]]:
        return {
            k: v.to_phlower_tensors(
                device=device,
                non_blocking=non_blocking,
                disable_dimensions=disable_dimensions,
            )
            for k, v in self._phlower_sequece_dict.items()
        }
