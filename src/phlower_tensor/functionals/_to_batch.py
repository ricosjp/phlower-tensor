from collections.abc import Sequence

from pipe import select

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._tensor import PhlowerTensor
from phlower_tensor.utils.enums import ConcatenateType

from ._check import is_same_layout
from ._concatenate import concatenate


def to_batch(
    tensors: Sequence[PhlowerTensor],
    dense_concat_dim: int | None = None,
    batch_mode: ConcatenateType | None = None,
) -> tuple[PhlowerTensor, GraphBatchInfo]:
    if not is_same_layout(tensors):
        raise ValueError(
            "The same layout tensors can be converted to batched tensor."
            "Several layouts are found in an argument."
        )

    batch_mode = batch_mode or ConcatenateType.auto_determine(
        tensors[0].is_sparse
    )
    if dense_concat_dim is None and batch_mode == ConcatenateType.axis:
        dense_concat_dim = 1 if tensors[0].is_time_series else 0

    concat_tensor = concatenate(
        tensors, dense_dim=dense_concat_dim, mode=batch_mode
    )
    # NOTE: After concatenation performed successfully,
    # the batch info can be created without checking the layout of tensors.
    batch_info = _create_batch_info(tensors, dense_concat_dim, batch_mode)

    return concat_tensor, batch_info


def _create_batch_info(
    tensors: Sequence[PhlowerTensor],
    dense_concat_dim: int | None,
    mode: ConcatenateType,
) -> GraphBatchInfo:
    _shapes = list(tensors | select(lambda x: x.shape))

    match mode:
        case ConcatenateType.block_diagonal:
            _sizes = list(tensors | select(lambda x: x.values().size()[0]))
            # NOTE: Assume dim=0 corresponds to n_nodes When sparse
            n_nodes = tuple(v[0] for v in _shapes)
            return GraphBatchInfo(_sizes, _shapes, n_nodes=n_nodes)

        case ConcatenateType.axis | ConcatenateType.index_shifting:
            dense_concat_dim = dense_concat_dim or 0
            _sizes = list(tensors | select(lambda x: x.size()))
            n_nodes = tuple(v[dense_concat_dim] for v in _shapes)

            return GraphBatchInfo(
                _sizes,
                _shapes,
                n_nodes=n_nodes,
                dense_concat_dim=dense_concat_dim,
            )

        case _:
            raise NotImplementedError(
                f"Batch mode {mode} is not implemented for batch info creation."
            )
