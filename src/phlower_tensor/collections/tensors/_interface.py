from __future__ import annotations

import abc
from collections.abc import (
    Callable,
    ItemsView,
    KeysView,
    Sequence,
)

import torch

from phlower_tensor._array import IPhlowerArray
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.utils.typing import ArrayDataType

_ComparableType = torch.Tensor | PhlowerTensor | float | int


class IPhlowerTensorCollections(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self, value: dict[str, torch.Tensor | PhlowerTensor]
    ) -> None: ...

    @abc.abstractmethod
    def __add__(
        self, __value: IPhlowerTensorCollections | _ComparableType
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __sub__(
        self, __value: IPhlowerTensorCollections | _ComparableType
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __mul__(
        self, __value: IPhlowerTensorCollections | _ComparableType
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __truediv__(
        self, __value: IPhlowerTensorCollections | _ComparableType
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __contains__(self, key: str): ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def slice(
        self, slice_range: tuple[slice, ...]
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __getitem__(self, key: str) -> PhlowerTensor: ...

    @abc.abstractmethod
    def min_len(self) -> int: ...

    @abc.abstractmethod
    def to_numpy(self) -> dict[str, ArrayDataType]: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @abc.abstractmethod
    def values(self): ...

    @abc.abstractmethod
    def items(self) -> ItemsView[str, PhlowerTensor]: ...

    @abc.abstractmethod
    def pop(self, key: str, default: PhlowerTensor | None = None): ...

    @abc.abstractmethod
    def sum(self, weights: dict[str, float] | None = None) -> PhlowerTensor: ...

    @abc.abstractmethod
    def mean(
        self, weights: dict[str, float] | None = None
    ) -> PhlowerTensor: ...

    @abc.abstractmethod
    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def unique_item(self) -> PhlowerTensor: ...

    @abc.abstractmethod
    def update(
        self, data: IPhlowerTensorCollections, overwrite: bool = False
    ) -> None: ...

    @abc.abstractmethod
    def mask(self, keys: list[str]) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def apply(self, function: Callable) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def clone(self) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def snapshot(self, time_index: int) -> IPhlowerTensorCollections:
        """
        Create a snapshot of the current state of the collection
          at a given time index. Timeseries tensor is converted to
          a tensor with the specified time index by index access.
          Non-timeseries tensors are returned as is.

        Args:
            time_index (int): The index of the time point.
        Returns:
            IPhlowerTensorCollections: A new collection containing
            the state of the tensors at the specified time index.
        """
        ...

    @abc.abstractmethod
    def to_phlower_arrays_dict(self) -> dict[str, IPhlowerArray]: ...

    @abc.abstractmethod
    def get_time_series_length(self) -> int:
        """
        Get the length of the time series in the collection.
        Returns:
            int: The length of the time series.
        """
        ...

    @abc.abstractmethod
    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> IPhlowerTensorCollections: ...
