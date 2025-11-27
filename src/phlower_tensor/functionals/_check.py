from collections.abc import Sequence

from pipe import select, uniq

from phlower_tensor._tensor import PhlowerDimensionTensor, PhlowerTensor


def is_same_layout(tensors: Sequence[PhlowerTensor]) -> bool:
    is_sparse_flags: set[bool] = set(tensors | select(lambda x: x.is_sparse))

    return len(is_sparse_flags) == 1


def is_same_dimensions(tensors: Sequence[PhlowerTensor]) -> bool:
    if len(tensors) == 0:
        return True

    dimensions: set[PhlowerDimensionTensor] = list(
        tensors
        | select(lambda x: x.dimension == tensors[0].dimension)  # type: ignore[assignment]
        | uniq
    )

    return len(dimensions) == 1
