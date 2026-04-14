import numpy as np
import pytest
import torch
from scipy import sparse as sp

from phlower_tensor import phlower_tensor
from phlower_tensor._array import phlower_array
from phlower_tensor.functionals._spmm import spmm


@pytest.mark.parametrize(
    "size, is_time_series, repeat",
    [
        ((10, 1), False, 1),
        ((10, 16), False, 1),
        ((10, 3, 16), False, 1),
        ((4, 10, 1), True, 1),
        ((4, 10, 16), True, 1),
        ((4, 10, 3, 16), True, 1),
        ((10, 1), False, 5),
        ((10, 16), False, 5),
        ((10, 3, 16), False, 5),
        ((4, 10, 1), True, 5),
        ((4, 10, 16), True, 5),
        ((4, 10, 3, 16), True, 5),
    ],
)
def test__spmm(size: tuple[int], is_time_series: bool, repeat: bool):
    _tensor = phlower_tensor(torch.rand(*size), is_time_series=is_time_series)
    n = _tensor.n_vertices()
    sparse = phlower_tensor(torch.rand(n, n).to_sparse())

    actual_spmm = spmm(sparse, _tensor, repeat=repeat)
    np_actual_spmm = actual_spmm.to_numpy()
    sp_sparse = sp.coo_array(sparse.to_tensor().to_dense().numpy())
    np_dense = _tensor.to_tensor().numpy()

    def assert_correct(actual: np.ndarray, array: np.ndarray):
        dim_feat = len(array.shape) - 1
        if dim_feat == 1:
            desired = array
            for _ in range(repeat):
                desired = sp_sparse @ desired
            norm = np.mean(np.linalg.norm(desired, axis=-1))
            np.testing.assert_almost_equal(
                actual / norm, desired / norm, decimal=5
            )
            return

        for i in range(array.shape[1]):
            assert_correct(actual[:, i], array[:, i])
        return

    if is_time_series:
        assert actual_spmm.is_time_series
        for t in range(size[0]):
            assert_correct(np_actual_spmm[t], np_dense[t])
    else:
        assert_correct(np_actual_spmm, np_dense)


@pytest.mark.parametrize(
    "size, sparse_size, is_time_series, desired_shape",
    [
        ((5, 10), (8, 5), False, (8, 10)),
        ((4, 10, 16), (5, 10), True, (4, 5, 16)),
    ],
)
def test__spmm_with_not_squared_sparse_matrix(
    size: tuple[int],
    sparse_size: tuple[int],
    is_time_series: bool,
    desired_shape: tuple[int],
):
    _tensor = phlower_tensor(torch.rand(*size), is_time_series=is_time_series)

    sparse_array = phlower_array(
        sp.random(*sparse_size, density=0.4, dtype=np.float32)
    )
    sparse_tensor = phlower_tensor(sparse_array.to_tensor())

    actual_spmm = spmm(sparse_tensor, _tensor, repeat=1)

    assert actual_spmm.shape == desired_shape


@pytest.mark.parametrize(
    "sparse_dimension, dense_dimension",
    [
        (None, None),
        ({}, {}),
        ({"L": -2}, {"L": 1, "T": -1}),
    ],
)
@pytest.mark.parametrize("repeat", [1, 2, 10])
def test__spmm_dimension_correct(
    sparse_dimension: dict[str, int] | None,
    dense_dimension: dict[str, int] | None,
    repeat: int,
):
    _tensor = phlower_tensor(torch.rand(10, 3, 1), dimension=dense_dimension)

    sparse_array = phlower_array(
        sp.random(10, 10, density=0.4, dtype=np.float32)
    )
    sparse_tensor = phlower_tensor(
        sparse_array.to_tensor(), dimension=sparse_dimension
    )

    actual_spmm = spmm(sparse_tensor, _tensor, repeat=repeat)

    if dense_dimension is None:
        assert actual_spmm.dimension is None
    else:
        assert (
            actual_spmm.dimension
            == sparse_tensor.dimension**repeat * _tensor.dimension
        )


@pytest.mark.parametrize("n_mat", [10, 100])
@pytest.mark.parametrize("n_feature", [1, 4])
def test__spmm_gradient_correct(n_mat: int, n_feature: int):
    dense = torch.rand(n_mat, n_feature)
    dense_for_phlower = phlower_tensor(dense.clone().requires_grad_(True))
    dense_for_torch = dense.clone().requires_grad_(True)

    sp_array = sp.random(n_mat, n_mat, density=0.4, dtype=np.float32)
    indices = torch.stack(
        [torch.from_numpy(sp_array.row), torch.from_numpy(sp_array.col)], dim=0
    )
    values = torch.from_numpy(sp_array.data.astype(np.float32))
    values_for_phlower = values.clone().requires_grad_(True)
    sparse_for_phlower = phlower_tensor(
        torch.sparse_coo_tensor(
            indices=indices, values=values_for_phlower, size=sp_array.shape
        )
    )
    values_for_torch = values.clone().requires_grad_(True)
    sparse_for_torch = torch.sparse_coo_tensor(
        indices=indices, values=values_for_torch, size=sp_array.shape
    )

    actual_loss = torch.sum(spmm(sparse_for_phlower, dense_for_phlower))
    actual_loss.backward()

    desired_loss = torch.sum(torch.sparse.mm(sparse_for_torch, dense_for_torch))
    desired_loss.backward()

    torch.testing.assert_close(
        dense_for_phlower.to_tensor().grad, dense_for_torch.grad
    )
    torch.testing.assert_close(values_for_phlower.grad, values_for_torch.grad)


@pytest.fixture
def temporarily_limit_cuda_memory():
    # Setup, limiting memory
    device = 0
    total_memory_bytes = torch.cuda.get_device_properties("cuda:0").total_memory
    limit_bytes = 100 * (1024**2)  # 100 MiB
    fraction = limit_bytes / total_memory_bytes

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)

    yield

    # Teardown, restoring memory
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0, device=device)


@pytest.mark.gpu_test
@pytest.mark.parametrize("n_mat", [10000])
@pytest.mark.parametrize("n_feature", [4])
def test__spmm_gradient_memory_efficient(
    n_mat: int, n_feature: int, temporarily_limit_cuda_memory: callable
):
    dense = torch.ones(n_mat, n_feature)
    dense_for_phlower = phlower_tensor(dense.clone().requires_grad_(True)).to(
        "cuda:0"
    )
    dense_for_torch = dense.clone().requires_grad_(True).to("cuda:0")

    sp_array = sp.eye(n_mat, dtype=np.float32).tocoo()
    indices = torch.stack(
        [torch.from_numpy(sp_array.row), torch.from_numpy(sp_array.col)], dim=0
    )
    values = torch.from_numpy(sp_array.data.astype(np.float32))
    values_for_phlower = values.clone().requires_grad_(True)
    sparse_for_phlower = phlower_tensor(
        torch.sparse_coo_tensor(
            indices=indices, values=values_for_phlower, size=sp_array.shape
        ).to("cuda:0")
    )
    values_for_torch = values.clone().requires_grad_(True)
    sparse_for_torch = torch.sparse_coo_tensor(
        indices=indices, values=values_for_torch, size=sp_array.shape
    ).to("cuda:0")

    # Test the phlower implementation can run backward
    torch.cuda.empty_cache()
    actual_loss = torch.sum(spmm(sparse_for_phlower, dense_for_phlower))
    actual_loss.backward()

    # Test the torch implementation fails due to the memory
    torch.cuda.empty_cache()
    desired_loss = torch.sum(torch.sparse.mm(sparse_for_torch, dense_for_torch))
    with pytest.raises(torch.OutOfMemoryError, match="CUDA out of memory"):
        desired_loss.backward()
