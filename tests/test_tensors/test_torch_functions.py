from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import assume, given
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
    zero_dimension: bool | st.SearchStrategy[bool] = False,
    disable_dimension: bool | st.SearchStrategy[bool] = False,
    allow_nan: bool = True,
    allow_infinity: bool = True,
) -> PhlowerTensor | list[PhlowerTensor]:
    _shape = draw(shape)

    zero_dimension = (
        draw(zero_dimension)
        if isinstance(zero_dimension, st.SearchStrategy)
        else zero_dimension
    )
    disable_dimension = (
        draw(disable_dimension)
        if isinstance(disable_dimension, st.SearchStrategy)
        else disable_dimension
    )

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
                    elements=st.floats(
                        allow_nan=allow_nan,
                        allow_infinity=allow_infinity,
                        width=32,
                    ),
                )
            )
        ),
        dimension=dimensions,
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda op, x: op(x), id="args"),
        pytest.param(lambda op, x: op(input=x), id="kwargs"),
    ],
)
@pytest.mark.parametrize(
    "func, numpy_func",
    [(torch.sin, np.sin), (torch.cos, np.cos), (torch.tanh, np.tanh)],
)
@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=True,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test__torch_functions_wo_dimension(
    call: Callable[..., PhlowerTensor],
    value: PhlowerTensor,
    func: Callable,
    numpy_func: Callable,
):
    actual = call(func, value)

    desired = torch.from_numpy(numpy_func(value.numpy()))
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda op, x: op(x), id="args"),
        pytest.param(lambda op, x: op(input=x), id="kwargs"),
    ],
)
@pytest.mark.parametrize("func", [torch.sin, torch.cos])
@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_functions_raise_error(
    call: Callable[..., PhlowerTensor], value: PhlowerTensor, func: Callable
):
    with pytest.raises(
        DimensionIncompatibleError, match="Should be dimensionless to apply"
    ):
        _ = call(func, value)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda condition, x, y: torch.where(condition, x, y), id="args"
        ),
        pytest.param(
            lambda condition, x, y: torch.where(condition, input=x, other=y),
            id="mixed",
        ),
        pytest.param(
            lambda condition, x, y: torch.where(
                condition=condition, input=x, other=y
            ),
            id="kwargs",
        ),
    ],
)
@given(
    value=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_where(call: Callable[..., PhlowerTensor], value: PhlowerTensor):
    scalar = phlower_tensor(torch.tensor(0.5), dimension=value.dimension)
    actual: PhlowerTensor = call(value > scalar, value, 0.0)

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
    "call",
    [
        pytest.param(lambda x, dim=None: squeeze(x, dim), id="args"),
        pytest.param(
            lambda x, dim=None: squeeze(x, dim=dim),
            id="mixed",
        ),
        pytest.param(
            lambda x, dim=None: squeeze(input=x, dim=dim),
            id="kwargs",
        ),
    ],
)
@pytest.mark.parametrize(
    "dim, shape, desired_shape",
    [
        (None, (3, 4, 1, 5), (3, 4, 5)),
        (None, (3, 1, 1, 5), (3, 5)),
        (0, (1, 3, 1, 4, 5), (3, 1, 4, 5)),
        (1, (3, 1, 1, 1, 5), (3, 1, 1, 5)),
        (2, (3, 4, 1, 5), (3, 4, 5)),
        (-1, (3, 4, 3, 1), (3, 4, 3)),
    ],
)
def test__torch_squeeze(
    call: Callable[..., PhlowerTensor],
    dim: int | None,
    shape: tuple[int, ...],
    desired_shape: tuple[int, ...],
):
    value = phlower_tensor(
        torch.rand(shape),
        dimension=[1.0 for _ in range(len(PhysicalDimensionSymbolType))],
    )
    if dim is not None:
        actual: PhlowerTensor = call(value, dim)
    else:
        actual: PhlowerTensor = call(value)

    desired = torch.rand(desired_shape)
    assert actual.shape == desired.shape
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(), value.dimension.numpy(), decimal=5
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda x, shifts, dims: torch.roll(x, shifts, dims), id="args"
        ),
        pytest.param(
            lambda x, shifts, dims: torch.roll(x, shifts=shifts, dims=dims),
            id="mixed",
        ),
        pytest.param(
            lambda x, shifts, dims: torch.roll(
                input=x, shifts=shifts, dims=dims
            ),
            id="kwargs",
        ),
    ],
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
    call: Callable[..., PhlowerTensor],
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

    actual: PhlowerTensor = call(value, shifts, dims)

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


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda x, dim, index: torch.index_select(x, dim, index), id="args"
        ),
        pytest.param(
            lambda x, dim, index: torch.index_select(x, dim=dim, index=index),
            id="mixed",
        ),
        pytest.param(
            lambda x, dim, index: torch.index_select(
                input=x, dim=dim, index=index
            ),
            id="kwargs",
        ),
    ],
)
@pytest.mark.parametrize(
    "shape, dim, index",
    [
        ((3, 4, 5), 0, [0, 2]),
        ((3, 4, 5), 1, [3, 0, 3]),
        ((3, 4, 5), -1, [4]),
    ],
)
@pytest.mark.parametrize("zero_dimension", [True, False])
@given(dimensions=random_dimensions())
def test__torch_index_select(
    call: Callable[..., PhlowerTensor],
    dimensions: list[float],
    shape: tuple[int, ...],
    dim: int,
    index: list[int],
    zero_dimension: bool,
):
    value = phlower_tensor(
        torch.rand(shape),
        dimension=None if zero_dimension else dimensions,
    )

    actual: PhlowerTensor = call(value, dim, torch.tensor(index))

    if zero_dimension:
        assert actual.dimension is None
    else:
        np.testing.assert_array_almost_equal(
            actual.dimension.numpy(),
            value.dimension.numpy(),
            decimal=5,
        )

    desired = torch.index_select(
        value.to_tensor(), dim=dim, index=torch.tensor(index)
    )
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# region torch.linalg.cross


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x, y: torch.linalg.cross(x, y, dim=-1), id="args"),
        pytest.param(
            lambda x, y: torch.linalg.cross(x, other=y, dim=-1), id="mixed"
        ),
        pytest.param(
            lambda x, y: torch.linalg.cross(input=x, other=y, dim=-1),
            id="kwargs",
        ),
    ],
)
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
def test__torch_linalg_cross(
    call: Callable[..., PhlowerTensor],
    tensor1: PhlowerTensor,
    tensor2: PhlowerTensor,
):
    actual = call(tensor1, tensor2)

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


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x, y: torch.linalg.cross(x, y, dim=-1), id="args"),
        pytest.param(
            lambda x, y: torch.linalg.cross(x, other=y, dim=-1), id="mixed"
        ),
        pytest.param(
            lambda x, y: torch.linalg.cross(input=x, other=y, dim=-1),
            id="kwargs",
        ),
    ],
)
@given(
    tensor=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=3, max_value=3),
        ),
        disable_dimension=True,
    )
)
def test__torch_linalg_cross_with_raw_tensor(
    call: Callable[..., PhlowerTensor], tensor: PhlowerTensor
):
    tensor1 = tensor
    tensor2 = torch.rand(tensor.shape)

    actual: PhlowerTensor = call(tensor1, tensor2)
    assert actual.dimension is None

    desired = torch.linalg.cross(tensor1.to_tensor(), tensor2, dim=-1)
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


# endregion

# region torch.linalg.norm, torch.linalg.vector_norm


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda op, x, ord, dim: op(x, ord, dim), id="args"),
        pytest.param(
            lambda op, x, ord, dim: op(x, ord=ord, dim=dim), id="mixed"
        ),
        pytest.param(
            lambda op, x, ord, dim: op(input=x, ord=ord, dim=dim), id="kwargs"
        ),
    ],
)
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
    call: Callable[..., PhlowerTensor],
    tensor: PhlowerTensor,
    func: Callable,
    ord: float,
    dim: int,
):
    actual = call(func, tensor, ord, dim)
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
    "call",
    [
        pytest.param(
            lambda x, lower, upper: torch.clamp(x, lower, upper), id="args"
        ),
        pytest.param(
            lambda x, lower, upper: torch.clamp(x, min=lower, max=upper),
            id="mixed",
        ),
        pytest.param(
            lambda x, lower, upper: torch.clamp(input=x, min=lower, max=upper),
            id="kwargs",
        ),
    ],
)
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
    call: Callable[..., PhlowerTensor],
    dimensions: list[float],
    values: list[float],
    min: float,
    max: float,
):
    tensor = phlower_tensor(torch.tensor(values), dimension=dimensions)
    actual = call(tensor, min, max)

    assert np.all(torch.min(actual).numpy() == min)
    assert np.all(torch.max(actual).numpy() == max)

    desired = torch.clamp(tensor.to_tensor(), min=min, max=max)

    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda x, lower, upper: torch.clamp(x, lower, upper), id="args"
        ),
        pytest.param(
            lambda x, lower, upper: torch.clamp(x, min=lower, max=upper),
            id="mixed",
        ),
        pytest.param(
            lambda x, lower, upper: torch.clamp(input=x, min=lower, max=upper),
            id="kwargs",
        ),
    ],
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
def test__torch_clamp(
    call: Callable[..., PhlowerTensor],
    value: PhlowerTensor,
    min_value: float,
    max_value: float,
):
    actual: PhlowerTensor = call(value, min_value, max_value)

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


# region torch_eq


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x, y: torch.eq(x, y), id="args"),
        pytest.param(lambda x, y: torch.eq(x, other=y), id="mixed"),
        pytest.param(lambda x, y: torch.eq(input=x, other=y), id="kwargs"),
    ],
)
@given(
    values=random_phlower_tensor_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=st.booleans(),
        disable_dimension=st.booleans(),
    )
)
def test__torch_eq(call: Callable[..., PhlowerTensor], values: PhlowerTensor):
    tensor1 = values
    tensor2 = phlower_tensor(
        values.to_tensor().detach().clone(), dimension=values.dimension
    )

    actual: PhlowerTensor = call(tensor1, tensor2)

    if tensor1.has_dimension:
        assert actual.dimension.is_dimensionless
    else:
        assert actual.dimension is None

    desired = torch.eq(tensor1.to_tensor(), tensor2.to_tensor())
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x, y: torch.eq(x, y), id="args"),
        pytest.param(lambda x, y: torch.eq(x, other=y), id="mixed"),
        pytest.param(lambda x, y: torch.eq(input=x, other=y), id="kwargs"),
    ],
)
@pytest.mark.parametrize(
    "dimension1, dimension2",
    [
        ({"L": 1.0}, {"L": 2.0}),
        ({"M": 1.0}, {"M": 2.0}),
        ({"T": 1.0}, {"T": 2.0, "L": 1.0}),
        ({"L": 1.0, "I": 1.0}, {"L": 1.0, "T": 1.0}),
    ],
)
def test__torch_eq_with_different_dimension(
    call: Callable[..., PhlowerTensor],
    dimension1: dict[str, float],
    dimension2: dict[str, float],
):
    tensor1 = phlower_tensor(
        torch.tensor([1.0, 2.0, 3.0]),
        dimension=dimension1,
    )
    tensor2 = phlower_tensor(
        torch.tensor([1.0, 2.0, 3.0]),
        dimension=dimension2,
    )

    with pytest.raises(
        DimensionIncompatibleError,
        match="Cannot compare tensors with different dimensions",
    ):
        _ = call(tensor1, tensor2)


# endregion


# region torch.linalg.cholesky


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: torch.linalg.cholesky(x), id="args"),
        pytest.param(lambda x: torch.linalg.cholesky(input=x), id="kwargs"),
    ],
)
@given(dimensions=random_dimensions())
def test__torch_cholesky(
    call: Callable[..., PhlowerTensor], dimensions: list[float]
):
    x = phlower_tensor(
        torch.tensor([[1.0, 0.5], [0.5, 1.0]]), dimension=dimensions
    )

    L = call(x)
    desired = torch.linalg.cholesky(x.to_tensor())

    np.testing.assert_array_almost_equal(L.numpy(), desired.numpy(), decimal=5)

    np.testing.assert_array_almost_equal(
        L.dimension.numpy(), x.dimension.numpy() * 0.5, decimal=3
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: torch.cholesky_inverse(x), id="args"),
        pytest.param(lambda x: torch.cholesky_inverse(input=x), id="kwargs"),
    ],
)
@given(dimensions=random_dimensions())
def test__torch_cholesky_inverse(
    call: Callable[..., PhlowerTensor], dimensions: list[float]
):
    # https://docs.pytorch.org/docs/stable/generated/torch.cholesky_inverse.html
    A = torch.randn(3, 3, dtype=torch.float32)
    A = A @ A.T + torch.eye(3, dtype=torch.float32) * 1e-3
    A = phlower_tensor(A, dimension=dimensions)
    L = torch.linalg.cholesky(A)

    Ainv = call(L)
    assert isinstance(Ainv, PhlowerTensor)

    desired = torch.cholesky_inverse(L.to_tensor())
    assert torch.dist(Ainv.to_tensor(), desired) < 1e-3

    np.testing.assert_array_almost_equal(
        Ainv.dimension.numpy(),
        A.dimension.numpy() * -1,
        decimal=3,
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda b, L: torch.cholesky_solve(b, L),
            id="args",
        ),
        pytest.param(
            lambda b, L: torch.cholesky_solve(b, input2=L),
            id="mixed",
        ),
        pytest.param(
            lambda b, L: torch.cholesky_solve(input=b, input2=L),
            id="kwargs",
        ),
    ],
)
@given(dimensions1=random_dimensions(), dimensions2=random_dimensions())
def test__torch_cholesky_solve(
    call: Callable[[PhlowerTensor, PhlowerTensor], PhlowerTensor],
    dimensions1: list[float],
    dimensions2: list[float],
):
    A = torch.randn(3, 3, dtype=torch.float32)
    A = A @ A.T + torch.eye(3, dtype=torch.float32) * 1e-3
    A = phlower_tensor(A, dimension=dimensions1)
    L = torch.linalg.cholesky(A)
    B = phlower_tensor(torch.randn(3, 2), dimension=dimensions2)

    X = call(B, L)
    assert isinstance(X, PhlowerTensor)

    desired = torch.cholesky_solve(B.to_tensor(), L.to_tensor())
    assert torch.dist(X.to_tensor(), desired) < 1e-3

    B_restored = torch.matmul(A, X)
    np.testing.assert_array_almost_equal(
        B_restored.numpy(), B.numpy(), decimal=3
    )

    np.testing.assert_array_almost_equal(
        B_restored.dimension.numpy(), B.dimension.numpy(), decimal=3
    )


# endregion


# region torch.linalg.inv


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: torch.linalg.inv(x), id="args"),
        pytest.param(lambda x: torch.linalg.inv(A=x), id="kwargs"),
    ],
)
@given(dimensions=random_dimensions())
def test__torch_inv(
    call: Callable[..., PhlowerTensor], dimensions: list[float]
):
    A = torch.randn(3, 3, dtype=torch.float32)
    A = A @ A.T + torch.eye(3, dtype=torch.float32) * 1e-3
    A = phlower_tensor(A, dimension=dimensions)

    Ainv = call(A)
    assert isinstance(Ainv, PhlowerTensor)

    desired = torch.linalg.inv(A.to_tensor())
    assert torch.dist(Ainv.to_tensor(), desired) < 1e-3

    np.testing.assert_array_almost_equal(
        Ainv.dimension.numpy(),
        A.dimension.numpy() * -1,
        decimal=3,
    )


# endregion


# region torch.linalg.pinv


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: torch.linalg.pinv(x), id="args"),
        pytest.param(lambda x: torch.linalg.pinv(input=x), id="kwargs"),
    ],
)
@given(dimensions=random_dimensions())
def test__torch_pinv(
    call: Callable[..., PhlowerTensor], dimensions: list[float]
):
    A = torch.randn(3, 10, dtype=torch.float32)
    A = A @ A.T
    A = phlower_tensor(A, dimension=dimensions)

    Apinv = call(A)
    assert isinstance(Apinv, PhlowerTensor)

    desired = torch.linalg.inv(A.to_tensor())
    assert torch.dist(Apinv.to_tensor(), desired) < 1e-3

    np.testing.assert_array_almost_equal(
        Apinv.dimension.numpy(),
        A.dimension.numpy() * -1,
        decimal=3,
    )


# endregion


# region torch.reciprocal


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: torch.reciprocal(x), id="args"),
        pytest.param(lambda x: torch.reciprocal(input=x), id="kwargs"),
    ],
)
@given(dimensions=random_dimensions())
def test__reciprocal(
    call: Callable[..., PhlowerTensor], dimensions: list[float]
):
    t = phlower_tensor(torch.rand(5) + 1, dimension=dimensions)
    t_inv = call(t)
    assert isinstance(t_inv, PhlowerTensor)
    assert (t_inv * t).dimension.is_dimensionless

    desired_dims_t = phlower_tensor(
        torch.rand(5), dimension=[-d for d in dimensions]
    )
    np.testing.assert_array_almost_equal(
        t_inv.dimension.numpy(), desired_dims_t.dimension.numpy(), decimal=5
    )


# endregion


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda op, x: op(x), id="args"),
        pytest.param(lambda op, x: op(input=x), id="kwargs"),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        torch.exp,
        torch.tanh,
        torch.sigmoid,
        torch.nn.functional.silu,
        torch.nn.functional.gelu,
    ],
)
@given(dimensions=random_dimensions())
def test__nonlinear_function_raises_when_not_dimensionless(
    call: Callable[..., PhlowerTensor], op: Callable, dimensions: list[float]
):
    assume(not all(d == 0.0 for d in dimensions))
    t = phlower_tensor(torch.rand(5), dimension=dimensions)
    with pytest.raises(
        DimensionIncompatibleError, match="Should be dimensionless to apply"
    ):
        call(op, t)
