from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as extra_np

from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.functionals import squeeze
from phlower_tensor.utils.enums import PhysicalDimensionSymbolType
from phlower_tensor.utils.exceptions import DimensionIncompatibleError


@st.composite
def random_dimensions(draw: Callable) -> list[float]:
    dimensions = draw(
        st.lists(
            elements=st.floats(allow_nan=False, allow_infinity=False, width=16),
            min_size=len(PhysicalDimensionSymbolType),
            max_size=len(PhysicalDimensionSymbolType),
        )
    )
    # To avoid zero dimension
    return [d + 1e-5 for d in dimensions]


@st.composite
def random_phlower_tensor_with_same_dimension_and_shape(
    draw: Callable,
    shape: tuple[int] | st.SearchStrategy[int],
    zero_dimension: bool = False,
    disable_dimension: bool = False,
) -> PhlowerTensor | list[PhlowerTensor]:
    _shape = draw(shape)

    if disable_dimension:
        dimensions = None
    elif zero_dimension:
        dimensions = {}
    else:
        dimensions = draw(
            st.lists(
                elements=st.floats(
                    allow_nan=False, allow_infinity=False, width=16
                ),
                min_size=len(PhysicalDimensionSymbolType),
                max_size=len(PhysicalDimensionSymbolType),
            )
        )
        # To avoid zero dimension
        dimensions = [d + 1e-5 for d in dimensions]

    return phlower_tensor(
        torch.from_numpy(
            draw(
                extra_np.arrays(
                    dtype=np.dtypes.Float32DType(),
                    shape=_shape,
                )
            )
        ),
        dimension=dimensions,
    )


@pytest.mark.parametrize("func", [torch.sin])
@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=True,
    )
)
def test__torch_functions(value: PhlowerTensor, func: Callable):
    actual = func(value)

    desired = torch.from_numpy(np.sin(value.numpy()))
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@pytest.mark.parametrize("func", [torch.sin])
@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_functions_raise_error(value: PhlowerTensor, func: Callable):
    with pytest.raises(
        DimensionIncompatibleError, match="Should be dimensionless to apply"
    ):
        _ = func(value)


@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_where(value: PhlowerTensor):
    scalar = phlower_tensor(torch.tensor(0.5), dimension=value.dimension)
    actual: PhlowerTensor = torch.where(value > scalar, value, 0.0)

    np_value = value.numpy()
    desired = torch.from_numpy(np.where(np_value > 0.5, np_value, 0.0))
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(),
        value.dimension.numpy(),
        decimal=5,
    )


@pytest.mark.parametrize(
    "dim, shape, desired_shape",
    [
        (None, (3, 4, 1, 5), (3, 4, 5)),
        (None, (3, 1, 1, 5), (3, 5)),
        (0, (1, 3, 1, 4, 5), (3, 1, 4, 5)),
        (1, (3, 1, 1, 1, 5), (3, 1, 1, 5)),
        (2, (3, 4, 1, 5), (3, 4, 5)),
    ],
)
def test__torch_squeeze(
    dim: int | None, shape: tuple[int, ...], desired_shape: tuple[int, ...]
):
    value = phlower_tensor(
        torch.rand(shape),
        dimension=[1.0 for _ in range(len(PhysicalDimensionSymbolType))],
    )
    if dim is not None:
        actual: PhlowerTensor = squeeze(value, dim=dim)
    else:
        actual: PhlowerTensor = squeeze(value)

    desired = torch.rand(desired_shape)
    assert actual.shape == desired.shape
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(), value.dimension.numpy(), decimal=5
    )


@pytest.mark.parametrize(
    "shape, shifts, dims",
    [
        ((3, 4, 5), 1, 0),
        ((3, 4, 5), -1, 1),
        ((3, 4, 5), 2, 2),
        ((3, 4, 5), (1, -2, 3), (0, 1, 2)),
    ],
)
@pytest.mark.parametrize("zero_dimension", [True, False])
@given(dimensions=random_dimensions())
def test__torch_roll(
    dimensions: list[float],
    shape: tuple[int, ...],
    shifts: int | tuple[int, ...],
    dims: int | tuple[int, ...],
    zero_dimension: bool,
):
    value = phlower_tensor(
        torch.rand(shape),
        dimension=None if zero_dimension else dimensions,
    )

    actual: PhlowerTensor = torch.roll(value, shifts=shifts, dims=dims)

    if zero_dimension:
        assert actual.dimension is None
    else:
        np.testing.assert_array_almost_equal(
            actual.dimension.numpy(),
            value.dimension.numpy(),
            decimal=5,
        )

    desired = torch.roll(value.to_tensor(), shifts=shifts, dims=dims)
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# region torch.linalg.cross


@given(
    tensor1=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.tuples(
            st.integers(min_value=10, max_value=10),
            st.integers(min_value=3, max_value=3),
        ),
        zero_dimension=st.booleans(),
    ),
    tensor2=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.tuples(
            st.integers(min_value=10, max_value=10),
            st.integers(min_value=3, max_value=3),
        ),
        zero_dimension=st.booleans(),
    ),
)
def test__torch_linalg_cross(tensor1: PhlowerTensor, tensor2: PhlowerTensor):
    actual = torch.linalg.cross(tensor1, tensor2, dim=-1)

    assert isinstance(actual, PhlowerTensor)
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(),
        tensor1.dimension.numpy() + tensor2.dimension.numpy(),
        decimal=5,
    )

    desired = torch.linalg.cross(
        tensor1.to_tensor(), tensor2.to_tensor(), dim=-1
    )
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@given(
    tensor=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=3, max_value=3),
        ),
        disable_dimension=st.booleans(),
    )
)
def test__torch_linalg_cross_with_raw_tensor(tensor: PhlowerTensor):
    tensor1 = tensor
    tensor2 = torch.rand(tensor.shape)

    actual: PhlowerTensor = torch.linalg.cross(tensor1, tensor2, dim=-1)
    assert actual.dimension is None

    desired = torch.linalg.cross(tensor1.to_tensor(), tensor2, dim=-1)
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# endregion

# region torch.linalg.norm, torch.linalg.vector_norm


@pytest.mark.parametrize("func", [torch.linalg.norm, torch.linalg.vector_norm])
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 1, -1])
@pytest.mark.parametrize("dim", [-1, 0, 1])
@given(
    tensor=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=1, max_value=10),
        ),
        zero_dimension=st.booleans(),
    )
)
def test__torch_linalg_norm(
    tensor: PhlowerTensor, func: Callable, ord: float, dim: int
):
    actual = func(tensor, ord=ord, dim=dim)
    assert isinstance(actual, PhlowerTensor)

    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(),
        tensor.dimension.numpy(),
        decimal=5,
    )

    desired = func(tensor.to_tensor(), ord=ord, dim=dim)
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# endregion


# region test torch.clamp


@pytest.mark.parametrize(
    "values, min, max",
    [
        ([-0.5, 0.2, 1.5, 2.0], 0.0, 1.0),
        ([-2.0, -1.0, 0.0, 0.5], -1.0, 0.0),
        ([0.0, 0.5, 1.0, 1.5], 0.5, 1.5),
    ],
)
@given(dimensions=random_dimensions())
def test__torch_clamp_with_explicit_cases(
    dimensions: list[float], values: list[float], min: float, max: float
):
    tensor = phlower_tensor(torch.tensor(values), dimension=dimensions)
    actual = torch.clamp(tensor, min=min, max=max)

    assert np.all(torch.min(actual).numpy() == min)
    assert np.all(torch.max(actual).numpy() == max)

    desired = torch.clamp(tensor.to_tensor(), min=min, max=max)

    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=st.booleans(),
        disable_dimension=st.booleans(),
    ),
    min_value=st.floats(-10.0, 0.0),
    max_value=st.floats(0.0, 10.0),
)
def test__torch_clamp(value: PhlowerTensor, min_value: float, max_value: float):
    actual: PhlowerTensor = torch.clamp(value, min=min_value, max=max_value)

    if value.has_dimension:
        np.testing.assert_array_almost_equal(
            actual.dimension.numpy(),
            value.dimension.numpy(),
            decimal=5,
        )
    else:
        assert actual.dimension is None

    desired = torch.clamp(value.to_tensor(), min=min_value, max=max_value)
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# endregion
