from collections.abc import Sequence
from typing import cast

import numpy as np
import torch

from phlower_tensor._tensor import (
    PhlowerTensor,
    phlower_tensor,
)
from phlower_tensor.utils.enums import ConcatenateType

from ._check import is_same_dimensions, is_same_layout


def concatenate(
    tensors: Sequence[PhlowerTensor],
    dense_dim: int | None = None,
    mode: ConcatenateType | None = None,
) -> PhlowerTensor:
    if not is_same_layout(tensors):
        raise ValueError("Cannot concatenate dense tensor and sparse tensor")

    if not is_same_dimensions(tensors):
        raise ValueError(
            "Cannot concatenate tensors which have different dimensions."
        )

    mode = mode or ConcatenateType.auto_determine(tensors[0].is_sparse)

    match mode:
        case ConcatenateType.axiswise:
            return _dense_concatenate(tensors, dim=dense_dim)
        case ConcatenateType.block_diagonal:
            assert dense_dim is None, (
                "Dense dimension should not be specified "
                "for block diagonal concatenation."
            )
            return _sparse_concatenate(tensors)
        case ConcatenateType.index_shifting:
            return _index_shifting_concatenate(tensors)
        case _:
            raise NotImplementedError(
                f"Concatenate mode {mode} is not implemented."
            )


def _dense_concatenate(
    tensors: Sequence[PhlowerTensor], dim: int | None = None
) -> PhlowerTensor:
    return torch.concatenate(tensors, dim=(dim or 0))  # type: ignore[call-overload]


def _sparse_concatenate(tensors: Sequence[PhlowerTensor]) -> PhlowerTensor:
    dimension = tensors[0].dimension

    offsets = torch.cumsum(
        torch.tensor(
            [
                tensors[i - 1].shape if i != 0 else (0, 0)
                for i in range(len(tensors))
            ]
        ),
        dim=0,
    )

    shape = np.sum([arr.shape for arr in tensors], axis=0)

    rows = torch.concatenate(
        [arr.indices()[0] + offsets[i, 0] for i, arr in enumerate(tensors)],
        dim=0,
    )
    cols = torch.concatenate(
        [arr.indices()[1] + offsets[i, 1] for i, arr in enumerate(tensors)],
        dim=0,
    )

    data = torch.concatenate([arr.values() for arr in tensors], dim=0)

    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]), values=data, size=tuple(shape)
    )
    return PhlowerTensor(sparse_tensor, dimension)


def _index_shifting_concatenate(
    tensors: Sequence[PhlowerTensor],
) -> PhlowerTensor:
    for v in tensors:
        _check_index_like_tensor(v.to_tensor())

    offsets = phlower_tensor(
        torch.cumsum(
            torch.tensor(
                [
                    tensors[i - 1].shape[0] if i != 0 else 0
                    for i in range(len(tensors))
                ]
            ),
            dim=0,
        ),
        dimension=tensors[0].dimension,
    )
    vals = torch.concatenate(
        [tensors[i] + offsets[i] for i in range(len(tensors))],  # type: ignore[misc]
        dim=0,
    )
    return cast(PhlowerTensor, vals)


def _check_index_like_tensor(tensor: torch.Tensor):
    if tensor.ndim != 1:
        raise ValueError(
            "Index shifting concatenation is only supported for 1D tensors."
        )

    if tensor.dtype not in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ]:
        raise ValueError(
            "Index shifting concatenation is only supported "
            "for integer tensors."
            f"Got {tensor.dtype} instead."
        )

    if (
        (torch.unique(tensor).shape[0] != tensor.shape[0])
        | (torch.min(tensor) != 0)
        | (torch.max(tensor) != tensor.shape[0] - 1)
    ):
        raise ValueError(
            "Index shifting concatenation is only supported for tensors "
            "which have unique values from 0 to n-1, "
            "where n is the number of elements."
        )
