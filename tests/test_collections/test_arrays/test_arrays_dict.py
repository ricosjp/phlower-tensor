from typing import NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp

from phlower_tensor import IPhlowerArray, PhlowerTensor, phlower_array
from phlower_tensor.collections import SequencedDictArray


class _DenseArrayInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: np.dtype
    index_like: bool = False


class _SparseArrayInfo(NamedTuple):
    dtype: np.dtype
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
            "b": _DenseArrayInfo(shape=(4,), dtype=np.float32),
        },
        {
            "x": _DenseArrayInfo(shape=(3, 1), dtype=np.float32),
            "y": _DenseArrayInfo(shape=(2, 2), dtype=np.float32),
            "z": _DenseArrayInfo(shape=(4,), dtype=np.float32),
        },
    ],
)
def test_names(
    n_nodes: list[int],
    name_to_info: dict[str, _DenseArrayInfo | _SparseArrayInfo],
):
    data = create_arrays(n_nodes, name_to_info)
    sequenced_dict_array = SequencedDictArray(data)

    assert set(sequenced_dict_array.get_names()) == set(name_to_info.keys())


@pytest.mark.parametrize("n_nodes", [[10, 20, 30], [5, 15], [8]])
@pytest.mark.parametrize(
    "name_to_info",
    [
        {
            "a": _DenseArrayInfo(shape=(3,), dtype=np.float32),
            "b": _DenseArrayInfo(shape=(4,), dtype=np.float32),
        },
        {
            "x": _DenseArrayInfo(shape=(3, 1), dtype=np.float32),
            "y": _SparseArrayInfo(dtype=np.float32),
            "z": _DenseArrayInfo(
                shape=(),
                dtype=np.int32,
                index_like=True,
            ),
        },
    ],
)
def test__to_phlower_tensors(
    n_nodes: list[int],
    name_to_info: dict[str, _DenseArrayInfo | _SparseArrayInfo],
):
    data = create_arrays(n_nodes, name_to_info)
    sequenced_dict_array = SequencedDictArray(data)

    values = sequenced_dict_array.to_phlower_tensors_dict(
        device="cpu",
        non_blocking=False,
        disable_dimensions=True,
    )

    assert set(values.keys()) == set(name_to_info.keys())

    for name, ph_tensors in values.items():
        should_sparse = isinstance(name_to_info[name], _SparseArrayInfo)
        for i, ph_tensor in enumerate(ph_tensors):
            assert isinstance(ph_tensor, PhlowerTensor)
            assert ph_tensor.is_sparse is should_sparse
            assert ph_tensor.shape[0] == n_nodes[i]
            assert (
                ph_tensor.to_tensor().to_dense().numpy().dtype
                == name_to_info[name].dtype
            )
