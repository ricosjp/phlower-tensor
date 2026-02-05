import torch

from phlower_tensor._base import PhysicalDimensions
from phlower_tensor._tensor._dimension._dimension_tensor import (
    PhlowerDimensionTensor,
    phlower_dimension_tensor,
    zero_dimension_tensor,
)

PhysicDimensionLikeObject = (
    PhysicalDimensions
    | PhlowerDimensionTensor
    | torch.Tensor
    | dict[str, float]
    | list[float]
    | tuple[float]
)


__all__ = [
    "PhlowerDimensionTensor",
    "PhysicDimensionLikeObject",
    "phlower_dimension_tensor",
    "zero_dimension_tensor",
]
