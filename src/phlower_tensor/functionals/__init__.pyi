from collections.abc import Sequence
from typing import overload

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._fields import SimulationField
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.collections.tensors._tensor_collections import (
    IPhlowerTensorCollections,
)

@overload
def unbatch(
    tensor: IPhlowerTensorCollections, *, n_nodes: list[int] | tuple[int]
) -> list[IPhlowerTensorCollections]: ...
@overload
def unbatch(
    tensor: PhlowerTensor, *, n_nodes: list[int] | tuple[int]
) -> list[PhlowerTensor]: ...
@overload
def unbatch(
    tensor: SimulationField, *, n_nodes: list[int] | tuple[int]
) -> list[SimulationField]: ...
@overload
def unbatch(
    tensor: IPhlowerTensorCollections, *, batch_info: GraphBatchInfo
) -> list[PhlowerTensor]: ...
@overload
def unbatch(
    tensor: PhlowerTensor, *, batch_info: GraphBatchInfo
) -> list[PhlowerTensor]: ...
@overload
def unbatch(
    tensor: SimulationField, *, batch_info: GraphBatchInfo
) -> list[PhlowerTensor]: ...
def to_batch(
    tensors: Sequence[PhlowerTensor], dense_concat_dim: int = 0
) -> tuple[PhlowerTensor, GraphBatchInfo]: ...
