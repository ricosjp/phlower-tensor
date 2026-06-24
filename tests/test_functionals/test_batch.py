from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from phlower_tensor import IPhlowerArray, PhlowerTensor, phlower_array
from phlower_tensor._tensor import (
    phlower_dimension_tensor,
    phlower_tensor,
)
from phlower_tensor.collections import SequencedDictArray
from phlower_tensor.functionals import to_batch
from phlower_tensor.utils.enums import ConcatenateType


@pytest.mark.parametrize(
    "shapes, dimensions, desired_shape",
    [
        ([(3, 5), (4, 7), (10, 1)], None, (17, 13)),
        (
            [(3, 5), (4, 7), (10, 1)],
            [
                {"M": 3, "L": 2},
                {"M": 3, "L": 2},
                {"M": 3, "L": 2},
            ],
            (17, 13),
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_mode",
    [
        None,
        ConcatenateType.block_diagonal,
    ],
)
def test__to_batch_for_sparse_tensors(
    shapes: list[tuple[int]],
    dimensions: dict | None,
    desired_shape: tuple[int],
    create_sparse_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[PhlowerTensor]
    ],
    batch_mode: ConcatenateType | None,
):
    tensors = create_sparse_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors, batch_mode=batch_mode)

    assert batched_tensor.is_sparse
    if dimensions is not None:
        assert batched_tensor.dimension == phlower_dimension_tensor(
            dimensions[0]
        )
    else:
        assert batched_tensor.dimension is None
    assert batched_tensor.shape == desired_shape
    assert batch_info.shapes == shapes


@pytest.mark.parametrize(
    "shapes, concat_dim, dimensions, desired_shape",
    [
        ([(3, 5), (4, 5), (10, 5)], 0, None, (17, 5)),
        ([(5, 3), (5, 4), (5, 10)], 1, None, (5, 17)),
        (
            [(6, 2), (5, 2), (11, 2)],
            0,
            [
                {"T": 3, "I": -1, "J": 2},
                {"T": 3, "I": -1, "J": 2},
                {"T": 3, "I": -1, "J": 2},
            ],
            (22, 2),
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_mode",
    [
        None,
        ConcatenateType.axiswise,
    ],
)
def test__to_batch_for_dense_tensors(
    shapes: list[tuple[int]],
    concat_dim: int,
    dimensions: dict,
    desired_shape: tuple[int],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[PhlowerTensor]
    ],
    batch_mode: ConcatenateType | None,
):
    tensors = create_dense_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors, concat_dim, batch_mode)

    assert not batched_tensor.is_sparse

    if dimensions is not None:
        assert batched_tensor.dimension == phlower_dimension_tensor(
            dimensions[0]
        )
    else:
        assert batched_tensor.dimension is None
    assert batched_tensor.shape == desired_shape
    assert batch_info.shapes == shapes


# region test for index shifting batch


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[0, 1, 2, 3], [0, 1, 2], [0, 1]], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ([[0, 1], [0, 2, 1], [3, 2, 0, 1]], [0, 1, 2, 4, 3, 8, 7, 5, 6]),
    ],
)
@pytest.mark.parametrize(
    "dimension",
    [
        None,
        {"M": 1},
        {"M": 1, "L": 2},
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
    ],
)
def test__values_with_index_shifting(
    values: list[list[int]],
    expected: list[list[int]],
    dimension: dict | None,
    dtype: torch.dtype,
):
    tensors = [
        phlower_tensor(v, dimension=dimension, dtype=dtype) for v in values
    ]
    batched, info = to_batch(tensors, batch_mode=ConcatenateType.index_shifting)

    batched_values = batched.to_tensor().tolist()
    assert batched_values == expected

    assert info.n_nodes == tuple(len(v) for v in values)
    assert info.shapes == [v.shape for v in tensors]


@pytest.mark.parametrize(
    "index_like_values",
    [
        [[0, 5, 2], [0, 1, 2]],  # max value is larger than n-1
        [[0, 1, 2], [10, 1, 2]],  # max value is larger than n-1
        [[-1, 1, 2], [1, 2, 3]],  # negative value
        [[0, 1, 2], [0, 1, 1]],  # duplicate value
        [[0, 1, 0], [0, 1, 2]],  # duplicate value
    ],
)
def test__failed_index_shifting_with_non_index_like_values(
    index_like_values: list[list[int]],
):
    tensors = [phlower_tensor(v, dtype=torch.int32) for v in index_like_values]

    with pytest.raises(
        ValueError, match="which have unique values from 0 to n-1"
    ):
        to_batch(tensors, batch_mode=ConcatenateType.index_shifting)


def test__failed_index_shifting_with_float():
    tensors = [
        phlower_tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        phlower_tensor([0.0, 1.0, 2.0], dtype=torch.float32),
    ]

    with pytest.raises(ValueError, match="only supported for integer tensors"):
        to_batch(tensors, batch_mode=ConcatenateType.index_shifting)


def test__failed_index_shifting_with_non_1d_tensor():
    tensors = [
        phlower_tensor([[0, 1], [2, 3]], dtype=torch.int32),
        phlower_tensor([[0, 1], [2, 3]], dtype=torch.int32),
    ]

    with pytest.raises(ValueError, match="only supported for 1D tensors"):
        to_batch(tensors, batch_mode=ConcatenateType.index_shifting)


# endregion


# region to_batch for SequencedDictArray


class _DenseArrayInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: np.dtype
    batch_mode: str | None = None
    index_like: bool = False


class _SparseArrayInfo(NamedTuple):
    dtype: np.dtype
    batch_mode: str | None = None
    shape: tuple[int, ...] = None


def create_arrays(
    n_nodes: list[int],
    name_to_info: dict[str, _DenseArrayInfo],
) -> list[dict[str, IPhlowerArray]]:
    data = [
        {
            name: phlower_array(_create_array(info, n_node))
            for name, info in name_to_info.items()
        }
        for n_node in n_nodes
    ]
    return data


def _create_array(
    info: _DenseArrayInfo | _SparseArrayInfo, n_node: int
) -> np.ndarray | sp.sparray:
    if isinstance(info, _DenseArrayInfo):
        if info.index_like:
            return np.arange(n_node, dtype=info.dtype)
        else:
            return np.random.rand(n_node, *info.shape).astype(info.dtype)
    if isinstance(info, _SparseArrayInfo):
        return sp.random(
            n_node,
            n_node,
            density=0.1,
            dtype=info.dtype,
        )

    raise ValueError(f"Unknown info type: {type(info)}")


@pytest.mark.parametrize("n_nodes", [[10, 20, 30], [5, 15], [8]])
@pytest.mark.parametrize(
    "name_to_info",
    [
        {
            "a": _DenseArrayInfo(shape=(3,), dtype=np.float32),
            "b": _DenseArrayInfo(
                shape=(4,), dtype=np.float32, batch_mode="axiswise"
            ),
        },
        {
            "x": _DenseArrayInfo(shape=(3, 1), dtype=np.float32),
            "y": _SparseArrayInfo(
                dtype=np.float32, batch_mode="block_diagonal"
            ),
            "z": _DenseArrayInfo(
                shape=(),
                dtype=np.int32,
                index_like=True,
                batch_mode="index_shifting",
            ),
        },
    ],
)
def test__batched_tensor(
    n_nodes: list[int],
    name_to_info: dict[str, _DenseArrayInfo | _SparseArrayInfo],
):
    data = create_arrays(n_nodes, name_to_info)
    sequenced_dict_array = SequencedDictArray(data)

    batch_mode_dict = {
        name: info.batch_mode for name, info in name_to_info.items()
    }
    values, infos = to_batch(
        sequenced_dict_array.to_phlower_tensors_dict(
            device="cpu",
            non_blocking=False,
            disable_dimensions=True,
        ),
        batch_mode_dict=batch_mode_dict,
    )

    for name, ph_tensor in values.items():
        should_sparse = isinstance(name_to_info[name], _SparseArrayInfo)
        assert ph_tensor.is_sparse is should_sparse
        assert ph_tensor.shape[0] == sum(n_nodes)
        assert (
            ph_tensor.to_tensor().to_dense().numpy().dtype
            == name_to_info[name].dtype
        )

    assert len(infos) == len(name_to_info)
    for _, info in infos.items():
        assert info.n_nodes == tuple(n_nodes)


# endregion
