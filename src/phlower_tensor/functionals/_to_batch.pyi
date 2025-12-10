from collections.abc import Sequence

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._tensor import PhlowerTensor

def to_batch(
    tensors: Sequence[PhlowerTensor], dense_concat_dim: int = 0
) -> tuple[PhlowerTensor, GraphBatchInfo]: ...
