from collections.abc import Sequence

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.utils.enums import ConcatenateType

def to_batch(
    tensors: Sequence[PhlowerTensor],
    dense_concat_dim: int | None = None,
    batch_mode: ConcatenateType | None = None,
) -> tuple[PhlowerTensor, GraphBatchInfo]: ...
