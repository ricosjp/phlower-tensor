from collections.abc import Callable

import pytest

from phlower_tensor._tensor import PhlowerTensor, phlower_dimension_tensor
from phlower_tensor.functionals import to_batch


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
def test__to_batch_for_sparse_tensors(
    shapes: list[tuple[int]],
    dimensions: dict | None,
    desired_shape: tuple[int],
    create_sparse_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[PhlowerTensor]
    ],
):
    tensors = create_sparse_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors)

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
def test__to_batch_for_dense_tensors(
    shapes: list[tuple[int]],
    concat_dim: int,
    dimensions: dict,
    desired_shape: tuple[int],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[PhlowerTensor]
    ],
):
    tensors = create_dense_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors, concat_dim)

    assert not batched_tensor.is_sparse

    if dimensions is not None:
        assert batched_tensor.dimension == phlower_dimension_tensor(
            dimensions[0]
        )
    else:
        assert batched_tensor.dimension is None
    assert batched_tensor.shape == desired_shape
    assert batch_info.shapes == shapes
