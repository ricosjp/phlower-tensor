from __future__ import annotations

import abc
from collections.abc import ItemsView, KeysView
from typing import Generic, TypeVar

import torch

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

T = TypeVar("T")


class ISimulationField(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, name: str) -> PhlowerTensor: ...

    @abc.abstractmethod
    def __contains__(self, name: str) -> bool: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @abc.abstractmethod
    def items(self) -> ItemsView[str, PhlowerTensor]: ...

    @abc.abstractmethod
    def get_batch_info(self, name: str) -> GraphBatchInfo: ...

    @abc.abstractmethod
    def get_batched_n_nodes(self, name: str) -> tuple[int]: ...

    @abc.abstractmethod
    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> ISimulationField[T]: ...

    @abc.abstractmethod
    def get_mesh(self) -> T: ...

    @abc.abstractmethod
    def overwrite(
        self, new_data: dict[str, PhlowerTensor] | IPhlowerTensorCollections
    ) -> ISimulationField: ...


class SimulationField(ISimulationField[None]):
    def __init__(
        self,
        field_tensors: IPhlowerTensorCollections | dict[str, PhlowerTensor],
        batch_info: dict[str, GraphBatchInfo] | None = None,
    ) -> None:
        if not isinstance(field_tensors, IPhlowerTensorCollections):
            field_tensors = phlower_tensor_collection(field_tensors)

        self._field_tensors = field_tensors

        if batch_info is None:
            batch_info = {}
        self._batch_info = batch_info

    def keys(self) -> KeysView[str]:
        return self._field_tensors.keys()

    def items(self) -> ItemsView[str, PhlowerTensor]:
        return self._field_tensors.items()

    def __getitem__(self, name: str) -> PhlowerTensor:
        if name not in self._field_tensors:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._field_tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._field_tensors

    def get_batch_info(self, name: str) -> GraphBatchInfo:
        if name not in self._batch_info:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._batch_info[name]

    def get_batched_n_nodes(self, name: str) -> tuple[int]:
        if not self._batch_info:
            raise ValueError("Information about batch is not found.")

        # NOTE: Assume that batch information is same among features.
        batch_info = self.get_batch_info(name)
        return batch_info.n_nodes

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> ISimulationField:
        return SimulationField(
            field_tensors=self._field_tensors.to(
                device, non_blocking=non_blocking
            ),
            batch_info=self._batch_info,
        )

    def get_mesh(self) -> None:
        return None

    def overwrite(
        self, new_data: dict[str, PhlowerTensor] | IPhlowerTensorCollections
    ) -> SimulationField:
        if not isinstance(new_data, IPhlowerTensorCollections):
            new_data = phlower_tensor_collection(new_data)

        # NOTE: Ensure that the override data
        # has the same shape as the original data.
        # Otherwise, it does not perform unbatching correctly.
        for k, v in new_data.items():
            if k in self._field_tensors:
                if self._field_tensors[k].shape != v.shape:
                    raise ValueError(
                        f"Shape mismatch for {k}: "
                        f"{self._field_tensors[k].shape} vs {v.shape}."
                        "Replacement data must have the same shape "
                        "as the original data."
                    )
        return SimulationField(
            field_tensors=self._field_tensors | new_data,
            batch_info=self._batch_info,
        )

    # HACK: Under construction
    # def calculate_laplacians(self, target: PhlowerTensor):
    #     ...

    # def calculate_gradient(self, target: PhlowerTensor):
    #     ...
