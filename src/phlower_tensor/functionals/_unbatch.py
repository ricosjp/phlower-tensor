from typing import TypeVar, cast

import torch

from phlower_tensor._batch import GraphBatchInfo
from phlower_tensor._fields import SimulationField
from phlower_tensor._tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections.tensors._tensor_collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

T = TypeVar("T", IPhlowerTensorCollections, SimulationField, PhlowerTensor)


def unbatch(
    tensor: T,
    batch_info: GraphBatchInfo | None = None,
    n_nodes: list[int] | tuple[int] | None = None,
) -> list[T]:
    if (not batch_info) and (not n_nodes):
        raise ValueError(
            "Parameters are missing. batch_info or n_nodes is necessary."
        )

    if isinstance(tensor, IPhlowerTensorCollections):
        return cast(list[T], _dict_unbatch(tensor, batch_info, n_nodes))

    if isinstance(tensor, SimulationField):
        return cast(list[T], _field_unbatch(tensor, batch_info, n_nodes))

    assert isinstance(tensor, PhlowerTensor)
    if tensor.is_sparse:
        if batch_info is None:
            raise ValueError(
                "batch_info is necessary when unbatching sparse tensor."
            )
        results = _sparse_unbatch(tensor.to_tensor(), batch_info)
        return cast(
            list[T], [phlower_tensor(v, tensor.dimension) for v in results]
        )

    # Only for dense tensor

    if batch_info:
        n_nodes = batch_info.n_nodes
        if not batch_info.is_concatenated:
            return [tensor]

    assert n_nodes is not None
    n_batch = len(n_nodes)
    if tensor.is_global(n_batch):
        n_global_nodes = [1] * n_batch
        results = _dense_unbatch(
            tensor.to_tensor(), n_global_nodes, tensor.shape_pattern.nodes_dim
        )
    elif tensor.is_voxel:
        if batch_info is None:
            raise ValueError(
                "batch_info is necessary when unbatching voxel tensor."
            )
        results = _dense_unbatch(
            tensor.to_tensor(), n_nodes, batch_info.dense_concat_dim
        )
    else:
        results = _dense_unbatch(
            tensor.to_tensor(), n_nodes, tensor.shape_pattern.nodes_dim
        )

    return cast(
        list[T],
        [
            phlower_tensor(
                v,
                dimension=tensor.dimension,
                is_time_series=tensor.is_time_series,
            )
            for v in results
        ],
    )


def _dict_unbatch(
    dict_tensor: IPhlowerTensorCollections,
    batch_info: GraphBatchInfo | None = None,
    n_nodes: list[int] | tuple[int] | None = None,
) -> list[IPhlowerTensorCollections]:
    if batch_info is None and n_nodes is None:
        raise ValueError(
            "Parameters are missing. batch_info or n_nodes is necessary."
        )
    if n_nodes is None:
        assert batch_info is not None
        n_nodes = batch_info.n_nodes
    n_batch = len(n_nodes)
    dict_unbatched = {
        k: unbatch(v, batch_info=batch_info, n_nodes=n_nodes)
        for k, v in dict_tensor.items()
    }
    return [
        phlower_tensor_collection(
            {k: v[i_batch] for k, v in dict_unbatched.items()}
        )
        for i_batch in range(n_batch)
    ]


def _field_unbatch(
    field_data: SimulationField,
    batch_info: GraphBatchInfo | None = None,
    n_nodes: list[int] | tuple[int] | None = None,
) -> list[SimulationField]:
    list_dict_unbatched = _dict_unbatch(
        field_data._field_tensors,
        batch_info,
        n_nodes=n_nodes,
    )
    return [
        SimulationField(dict_unbatched, batch_info=None)
        for dict_unbatched in list_dict_unbatched
    ]


def _sparse_unbatch(
    sparse_tensor: torch.Tensor, batch_info: GraphBatchInfo
) -> list[torch.Tensor]:
    sizes = torch.tensor(batch_info.sizes, dtype=torch.int32)
    offsets = torch.tensor(
        torch.cumsum(
            torch.tensor([[0, 0]] + batch_info.shapes[:-1], dtype=torch.int32),
            dim=0,
            dtype=torch.int32,
        )
    )
    sparse_tensor = sparse_tensor.coalesce()

    rows = sparse_tensor.indices()[0] - offsets[:, 0].repeat_interleave(sizes)
    rows = rows.split(batch_info.sizes)

    cols = sparse_tensor.indices()[1] - offsets[:, 1].repeat_interleave(sizes)
    cols = cols.split(batch_info.sizes)

    data = sparse_tensor.values().split(batch_info.sizes)

    results = [
        torch.sparse_coo_tensor(
            indices=torch.stack([rows[i], cols[i]]),
            values=data[i],
            size=batch_info.shapes[i],
        )
        for i in range(len(batch_info))
    ]
    return results


def _dense_unbatch(
    tensor: torch.Tensor, n_nodes: list[int] | tuple[int], node_dim: int | None
) -> list[torch.Tensor]:
    return tensor.split(n_nodes, dim=node_dim)
