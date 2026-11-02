from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch
from pipe import uniq

from phlower_tensor._base import PhysicalDimensions
from phlower_tensor.utils.enums import PhysicalDimensionSymbolType
from phlower_tensor.utils.exceptions import DimensionIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


def phlower_dimension_tensor(
    values: dict[str, float] | PhysicalDimensions,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> PhlowerDimensionTensor:
    """Create PhlowerDimensionTensor from dict or PhysicalDimensions object

    Args:
        values (dict[str, float] | PhysicalDimensions): values
        dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        device (str | torch.device, optional): device. Defaults to None.

    Returns:
        PhlowerDimensionTensor: PhlowerDimensionTensor object
    """
    if not isinstance(values, PhysicalDimensions):
        values = PhysicalDimensions(values)

    _list = values.to_list()
    return PhlowerDimensionTensor.from_list(_list, dtype=dtype).to(device)


def zero_dimension_tensor(
    device: str | torch.device | None,
) -> PhlowerDimensionTensor:
    zerodim = phlower_dimension_tensor({})
    if device is None:
        return zerodim

    return zerodim.to(device)


class PhlowerDimensionTensor:
    """
    PhlowerDimensionTensor is a tensor that represents the physical dimensions

    Examples
    --------
    >>> dimension_tensor = PhlowerDimensionTensor(
    ...     values={"M": 1.0, "L": 1.0, "T": -2.0}
    ... )
    """

    @classmethod
    def from_list(
        cls,
        values: list[float] | tuple[float],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> PhlowerDimensionTensor:
        """
        Parse from list object

        Args:
            values (list[float] | tuple[float]): list or tuple
                if length of values is not equal to the number of registered
                dimension type, raise ValueError.

        Returns:
            PhlowerDimensionTensor: tensor object
        """
        if len(values) != len(PhysicalDimensionSymbolType):
            raise ValueError(
                "length of values is not equal to the number of "
                f"registered dimension types. input: {len(values)}"
            )
        _tensor = (
            torch.tensor(values, dtype=dtype)
            .reshape(len(PhysicalDimensionSymbolType), 1)
            .to(device=device)
        )
        return PhlowerDimensionTensor(_tensor)

    def __init__(
        self,
        tensor: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if tensor is None:
            self._tensor = torch.zeros(
                (len(PhysicalDimensionSymbolType), 1), dtype=dtype
            )
            return

        assert isinstance(tensor, torch.Tensor), f"Unexpected type: {tensor}"

        self._tensor = tensor
        assert self._tensor.shape[0] == len(PhysicalDimensionSymbolType)

    def __sub__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.sub(self, __value))  # type: ignore[call-overload]

    def __rsub__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.rsub(self, __value))  # type: ignore[call-overload]

    def __add__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.add(self, __value))  # type: ignore[call-overload]

    def __radd__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.add(__value, self))  # type: ignore[call-overload]

    def __mul__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.mul(self, __value))  # type: ignore[arg-type]

    def __rmul__(self, __value: object):
        return cast(PhlowerDimensionTensor, torch.mul(__value, self))  # type: ignore[arg-type]

    def __truediv__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.div(self, __value))  # type: ignore[arg-type]

    def __rtruediv__(
        self, __value: PhlowerDimensionTensor | torch.Tensor | float | int
    ):
        return cast(PhlowerDimensionTensor, torch.div(__value, self))  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhlowerDimensionTensor):
            return NotImplemented

        return bool(torch.all(self._tensor == other._tensor).item())

    def __pow__(self, other: float | int) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor * other)

    def __repr__(self) -> str:
        texts = ", ".join(
            [
                f"{unit_type.name}: {self._tensor[unit_type.value, 0].item()}"
                for unit_type in PhysicalDimensionSymbolType
            ]
        )
        return f"{self.__class__.__name__}({texts})"

    def __matmul__(
        self, other: PhlowerDimensionTensor | torch.Tensor
    ) -> PhlowerDimensionTensor:
        return cast(PhlowerDimensionTensor, torch.matmul(self, other))  # type: ignore[arg-type]

    def __rmatmul__(
        self, other: PhlowerDimensionTensor | torch.Tensor
    ) -> PhlowerDimensionTensor:
        return cast(PhlowerDimensionTensor, torch.matmul(other, self))  # type: ignore[arg-type]

    def to_physics_dimension(self) -> PhysicalDimensions:
        """Convert to PhysicalDimensions object

        Returns:
            PhysicalDimensions: PhysicalDimensions object
        """
        _dict = self.to_dict()
        return PhysicalDimensions(_dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to dict

        Returns:
            dict[str, float]: Dict
        """
        return {
            k: self._tensor[v.value].cpu().numpy().item()
            for k, v in PhysicalDimensionSymbolType.__members__.items()
        }

    def to(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        dtype: torch.dtype | None = None,
    ) -> PhlowerDimensionTensor:
        """Convert to different device or data type

        Args:
            device (str | torch.device, optional): device. Defaults to None.
            non_blocking (bool, optional): non-blocking. Defaults to False.
            dtype (torch.dtype, optional): data type. Defaults to None.
        """
        new_dimension = self._tensor.to(
            device, non_blocking=non_blocking, dtype=dtype
        )
        new_dtype = dtype or self.dtype
        return PhlowerDimensionTensor(new_dimension, dtype=new_dtype)

    def detach(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.detach())

    def clone(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.clone())

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def is_dimensionless(self) -> bool:
        """Return True if the tensor is dimensionless.

        Returns:
            bool: True if the tensor is dimensionless.
        """
        return bool(torch.sum(torch.abs(self._tensor)) < 1e-5)

    def numpy(self) -> np.ndarray:
        """Convert to numpy array

        Returns:
            np.ndarray: numpy array
        """
        return self._tensor.detach().numpy()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ) -> PhlowerDimensionTensor:
        if kwargs is None:
            kwargs = {}

        if func.__name__ not in _HANDLED_FUNCTIONS:
            return NotImplemented

        return _HANDLED_FUNCTIONS[func.__name__](*args, **kwargs)


def dimension_wrap_implements(torch_function: Callable) -> Callable:
    """Register a torch function override for PhysicsUnitTensor"""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function.__name__] = func
        return func

    return decorator


@dimension_wrap_implements(torch.mean)
def mean(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


def _determine_device(
    *args: PhlowerDimensionTensor | torch.Tensor | float | int,
) -> torch.device:
    devices = {
        v.device
        for v in args
        if isinstance(v, PhlowerDimensionTensor | torch.Tensor)
    }

    if len(devices) != 1:
        raise ValueError(
            f"Cannot determine unique device. {devices}. args: {args}"
        )

    return devices.pop()


def _convert_phlower_dimension_tensors(
    *args: PhlowerDimensionTensor | torch.Tensor | float | int,
    device: str | torch.device | None = None,
) -> tuple[PhlowerDimensionTensor, ...]:
    _dimensions = tuple(
        v
        if isinstance(v, PhlowerDimensionTensor)
        else zero_dimension_tensor(device=device)
        for v in args
    )
    return _dimensions


@dimension_wrap_implements(torch.add)
def add(
    input: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    input, other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    if input != other:
        raise DimensionIncompatibleError(
            "Add operation for different physical dimensions is not allowed."
        )

    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.sub)
def sub(
    input: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    input, other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    if input != other:
        raise DimensionIncompatibleError(
            "Sub operation for different physical dimensions is not allowed."
        )

    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.rsub)
def rsub(
    input: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    input, other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    if input != other:
        raise DimensionIncompatibleError(
            "Sub operation for different physical dimensions is not allowed."
        )

    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.pow)
def pow(
    input: PhlowerDimensionTensor, exponent: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor * exponent)


@dimension_wrap_implements(torch.sqrt)
def sqrt(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor / 2)


@dimension_wrap_implements(torch.mul)
def mul(
    input: PhlowerDimensionTensor | torch.Tensor,
    other: PhlowerDimensionTensor | torch.Tensor,
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor + _other._tensor)


@dimension_wrap_implements(torch.div)
def div(
    input: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor - _other._tensor)


@dimension_wrap_implements(torch.reshape)
def reshape(
    input: PhlowerDimensionTensor, shape: tuple[int]
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.cat)
def cat(
    tensors: tuple[PhlowerDimensionTensor | torch.Tensor, ...],
    dim: int = 0,
    *,
    out: PhlowerDimensionTensor | None = None,
) -> PhlowerDimensionTensor:
    device = _determine_device(*tensors)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*tensors, device=device) | uniq
    )

    if len(uniq_inputs) != 1:
        raise DimensionIncompatibleError(
            "Only same physical dimensions are allowed when torch.cat."
        )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


@dimension_wrap_implements(torch.sparse.mm)
def sparse_mm(
    sparse: PhlowerDimensionTensor, dense: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in (sparse, dense)):
        return PhlowerDimensionTensor(sparse._tensor + dense._tensor)

    raise DimensionIncompatibleError(f"input1: {sparse}, input2: {dense}")


@dimension_wrap_implements(torch.nn.functional.linear)
def nn_linear(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.nn.functional.dropout)
def dropout(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.stack)
def stack(
    tensors: Sequence[PhlowerDimensionTensor], *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    device = _determine_device(*tensors)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*tensors, device=device) | uniq
    )

    if len(uniq_inputs) != 1:
        raise DimensionIncompatibleError(
            "Only same physical dimensions are allowed when torch.stack"
        )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


@dimension_wrap_implements(torch.max)
def max(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.min)
def min(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.linalg.norm)
def norm(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.linalg.vector_norm)
def _vector_norm(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.nn.functional.mse_loss)
def mse_loss(
    input: PhlowerDimensionTensor,
    target: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    if input != target:
        raise DimensionIncompatibleError()

    return input**2


@dimension_wrap_implements(torch.sum)
def _sum(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    if isinstance(input, PhlowerDimensionTensor):
        return input

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.concatenate)
def concatenate(
    tensors: Sequence[PhlowerDimensionTensor], *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    device = _determine_device(*tensors)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*tensors, device=device) | uniq
    )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


@dimension_wrap_implements(torch.exp)
def exp(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not input.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply exp but {input}"
        )
    return input


@dimension_wrap_implements(torch.tanh)
def tanh(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not input.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply tanh but {input}"
        )
    return input


@dimension_wrap_implements(torch.sigmoid)
def sigmoid(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not input.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply sigmoid but {input}"
        )
    return input


@dimension_wrap_implements(torch.nn.functional.leaky_relu)
def leaky_relu(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    # NOTE: Allow leaky relu operation also for dimensioned tensor
    #       because it is scale equivariant
    return input


@dimension_wrap_implements(torch.reciprocal)
def reciprocal(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(-input._tensor)


@dimension_wrap_implements(torch.unsqueeze)
def unsqueeze(
    input: PhlowerDimensionTensor, dim: int
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.abs)
def torch_abs(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.gt)
def torch_gt(
    input: PhlowerDimensionTensor,
    other: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    if input != other:
        raise DimensionIncompatibleError()
    return input


@dimension_wrap_implements(torch.ge)
def torch_ge(
    input: PhlowerDimensionTensor,
    other: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    if input != other:
        raise DimensionIncompatibleError()
    return input


@dimension_wrap_implements(torch.lt)
def torch_lt(
    input: PhlowerDimensionTensor,
    other: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    if input != other:
        raise DimensionIncompatibleError()
    return input


@dimension_wrap_implements(torch.le)
def torch_le(
    input: PhlowerDimensionTensor,
    other: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    if input != other:
        raise DimensionIncompatibleError()
    return input


@dimension_wrap_implements(torch.nn.functional.pad)
def _torch_pad(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.conv1d)
def _torch_conv1d(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.relu)
def _torch_relu(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    # NOTE: Allow relu operation also for dimensioned tensor
    #       because it is scale equivariant
    return input


@dimension_wrap_implements(torch.squeeze)
def _torch_squeeze(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.where)
def _torch_where(
    condition: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return condition


@dimension_wrap_implements(torch.sin)
def _torch_sin(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not input.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply sin but {input}"
        )
    return input


@dimension_wrap_implements(torch.cos)
def _torch_cos(input: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not input.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply cos but {input}"
        )
    return input


@dimension_wrap_implements(torch.matmul)
def _torch_matmul(
    input: PhlowerDimensionTensor | torch.Tensor,
    other: PhlowerDimensionTensor | torch.Tensor,
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor + _other._tensor)


@dimension_wrap_implements(torch.median)
def _torch_median(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.roll)
def _torch_roll(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.index_select)
def _torch_index_select(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor)


@dimension_wrap_implements(torch.linalg.cross)
def _torch_linalg_cross(
    input: PhlowerDimensionTensor | torch.Tensor,
    other: PhlowerDimensionTensor | torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor + _other._tensor)


@dimension_wrap_implements(torch.eq)
def _torch_eq(
    input: PhlowerDimensionTensor,
    other: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    device = _determine_device(input, other)
    input, other = _convert_phlower_dimension_tensors(
        input, other, device=device
    )
    if input != other:
        raise DimensionIncompatibleError(
            "Cannot compare tensors with different dimensions."
            f" input: {input}, other: {other}"
        )
    return zero_dimension_tensor(device=device)


@dimension_wrap_implements(torch.clamp)
def _torch_clamp(
    input: PhlowerDimensionTensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    return input


@dimension_wrap_implements(torch.linalg.cholesky)
def _torch_cholesky(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor / 2.0)


@dimension_wrap_implements(torch.cholesky_inverse)
def _torch_cholesky_inverse(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor * -2.0)


@dimension_wrap_implements(torch.cholesky_solve)
def _torch_cholesky_solve(
    input: PhlowerDimensionTensor | torch.Tensor,
    input2: PhlowerDimensionTensor | torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> PhlowerDimensionTensor:
    # input: B
    # input2: L, A = L @ L.T
    # output: A^-1 @ B = L^-T @ L^-1 @ B

    device = _determine_device(input, input2)
    B, L = _convert_phlower_dimension_tensors(input, input2, device=device)
    return PhlowerDimensionTensor(B._tensor - L._tensor * 2.0)


@dimension_wrap_implements(torch.linalg.inv)
def _torch_inv(
    A: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(A._tensor * -1.0)


@dimension_wrap_implements(torch.linalg.pinv)
def _pinv(
    input: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(input._tensor * -1.0)
