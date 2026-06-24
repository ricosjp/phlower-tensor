from collections.abc import Sequence
from typing import overload

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.utils.enums import ConcatenateType

@overload
def to_batch(
    tensors: Sequence[PhlowerTensor],
    dense_concat_dim: int | None = None,
    batch_mode: ConcatenateType | None = None,
) -> tuple[PhlowerTensor, GraphBatchInfo]: ...
@overload
def to_batch(
    tensors: dict[str, Sequence[PhlowerTensor]],
    batch_mode_dict: dict[str, ConcatenateType | None] | None = None,
) -> tuple[dict[str, PhlowerTensor], dict[str, GraphBatchInfo]]: ...
