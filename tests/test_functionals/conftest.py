from collections.abc import Callable

import numpy as np
import pytest
import torch
from scipy import sparse as sp

from phlower_tensor._array import phlower_array
from phlower_tensor._base import PhysicalDimensions
from phlower_tensor._tensor import PhlowerTensor, phlower_tensor


@pytest.fixture
def create_sparse_tensors() -> Callable[
    [list[tuple[int]], list[dict[str, float]] | None], list[PhlowerTensor]
]:
    def _create(
        shapes: list[tuple], dimensions: list[dict] | None = None
    ) -> list[PhlowerTensor]:
        rng = np.random.default_rng()
        if dimensions is None:
            return [
                phlower_tensor(
                    phlower_array(
                        sp.random(
                            shape[0], shape[1], density=0.1, random_state=rng
                        ),
                    ).to_tensor()
                )
                for shape in shapes
            ]

        return [
            phlower_tensor(
                phlower_array(
                    sp.random(
                        shape[0], shape[1], density=0.1, random_state=rng
                    ),
                ).to_tensor(),
                dimension=PhysicalDimensions(dims),
            )
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create


@pytest.fixture
def create_dense_tensors() -> Callable[
    [list[tuple[int]], list[dict[str, float]] | None, bool],
    list[PhlowerTensor],
]:
    def _create(
        shapes: list[tuple],
        dimensions: list[dict] | None = None,
        is_time_series: bool = False,
    ) -> list[PhlowerTensor]:
        if dimensions is None:
            return [
                phlower_tensor(
                    np.random.rand(*shape),
                    is_time_series=is_time_series,
                    dtype=torch.float32,
                )
                for shape in shapes
            ]

        return [
            phlower_tensor(
                np.random.rand(*shape),
                dimension=dims,
                is_time_series=is_time_series,
                dtype=torch.float32,
            )
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create
