import numpy as np
import pytest

from phlower_tensor._array import phlower_array


@pytest.mark.parametrize("componentwise", [True, False])
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 3, 1)])
def test__apply_function_when_skip_nan_is_False(
    componentwise: bool, shape: tuple[int, ...]
):
    arr = np.random.rand(*shape)
    random_indices = np.random.choice(arr.size, size=5, replace=False)
    arr.flat[random_indices] = np.nan

    ph_array = phlower_array(arr)

    ph_array_applied = ph_array.apply(
        function=lambda x: x * 2, componentwise=False, skip_nan=False
    )

    desired = arr * 2
    np.testing.assert_array_almost_equal(ph_array_applied.to_numpy(), desired)


def test__apply_function_when_skip_nan_is_True():
    arr = np.random.rand(10, 3, 1)
    random_indices = np.random.choice(arr.size, size=5, replace=False)
    arr.flat[random_indices] = np.nan

    ph_array = phlower_array(arr)

    with pytest.raises(ValueError, match="skip_nan=True is not supported"):
        _ = ph_array.apply(
            function=lambda x: x * 2, componentwise=False, skip_nan=True
        )


@pytest.mark.parametrize("componentwise", [True, False])
def test__reshape_when_skip_nan_is_True(componentwise: bool):
    arr = np.random.rand(10, 3, 1)
    random_indices = np.random.choice(arr.size, size=5, replace=False)
    arr.flat[random_indices] = np.nan

    assert np.isnan(arr).any()

    ph_array = phlower_array(arr)
    reshaped_wo_nan = ph_array.reshape(
        componentwise=componentwise, skip_nan=True
    )

    assert np.all(~np.isnan(reshaped_wo_nan.to_numpy()))


@pytest.mark.parametrize(
    "componentwise, shape, expected",
    [
        (True, (2, 3, 4), (6, 4)),
        (False, (2, 3, 4), (24, 1)),
        (True, (5, 3, 1), (15, 1)),
        (False, (5, 3, 1), (15, 1)),
    ],
)
def test__shape_after_reshape(
    componentwise: bool, shape: tuple[int, ...], expected: tuple[int, ...]
):
    arr = np.random.rand(*shape)
    ph_array = phlower_array(arr)

    reshaped = ph_array.reshape(componentwise=componentwise)

    assert reshaped.shape == expected
