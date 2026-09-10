from collections.abc import Callable
from typing import Protocol

import torch
from torch.autograd.function import FunctionCtx

from phlower_tensor._tensor import PhlowerTensor, phlower_tensor


def spmm(
    sparse: PhlowerTensor, x: PhlowerTensor, repeat: int = 1
) -> PhlowerTensor:
    """
    Computes sparse matrix times dense tensor along with the vertex axis.

    Args:
        sparse : PhlowerTensor:
            Sparse tensor.
        x : PhlowerTensor
            Dense tensor.
        repeat : int, optional
            The number of repetitions for multiplication. The default is 1.
    Returns:
        PhlowerTensor:
            Resultant tensor.
    """
    h, resultant_pattern = x.to_vertexwise()
    restore_pattern = f"{resultant_pattern} -> {x.shape_pattern.get_pattern()}"
    restore_axes_length = x.shape_pattern.get_pattern_to_size(drop_last=True)
    if (
        symbol := x.shape_pattern.n_nodes_pattern_symbol
    ) in restore_axes_length:
        restore_axes_length[symbol] = sparse.shape[0]

    for _ in range(repeat):
        if h.dimension is None:
            out_dimension = None
        else:
            out_dimension = sparse.dimension * h.dimension
        h = phlower_tensor(
            _SparseGradSpMM.apply(sparse.to_tensor(), h.to_tensor()),
            dimension=out_dimension,
        )
    return h.rearrange(restore_pattern, **restore_axes_length)


def _copy_doc_decorator(source: Callable) -> Callable:
    def decorator(target: Callable) -> Callable:
        target.__doc__ = source.__doc__
        return target

    return decorator


class FunctionCtxProtocol(Protocol):
    """
    Protocol for the context object used in the custom autograd function.
    This is a helper class to provide type hints and pass static type checking.
    """

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    @property
    def spmat_size(self) -> torch.Size: ...

    @spmat_size.setter
    def spmat_size(self, value: torch.Size) -> None: ...

    @_copy_doc_decorator(FunctionCtx.save_for_backward)
    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _SparseGradSpMM(torch.autograd.Function):
    clear_saved_tensors_on_access = True

    @staticmethod
    def forward(
        spmat: torch.Tensor,
        dense_x: torch.Tensor,
    ) -> torch.Tensor:
        spmat = spmat.coalesce()
        spmm = torch.sparse.mm(spmat, dense_x)
        return spmm

    @staticmethod
    def setup_context(
        ctx: FunctionCtxProtocol,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        spmat, dense_x = inputs
        spmat = spmat.coalesce()
        ctx.save_for_backward(
            spmat.indices(),
            spmat.values(),
            dense_x,
        )
        ctx.spmat_size = spmat.size()

    @staticmethod
    def backward(
        ctx: FunctionCtxProtocol, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        indices, values, dense_x = ctx.saved_tensors
        size = ctx.spmat_size

        # Gradient w.r.t. the learnable sparse values (A)
        if ctx.needs_input_grad[0]:
            # d L / d a_{ij} = (d L / d y_{ik}) (x^T)_{kj}
            row, col = indices[0], indices[1]
            grad_values = (grad_output[row] * dense_x[col]).sum(dim=-1)
            t_indices = torch.stack([indices[1], indices[0]], dim=0)
            grad_sparse = torch.sparse_coo_tensor(
                indices, grad_values, size=(size[0], size[1])
            )
        else:
            grad_sparse = None

        # Gradient w.r.t. the dense input (X)
        if ctx.needs_input_grad[1]:
            # d L / d x_{jk} = (a^T)_{ji} (d L / d y_{ik})
            t_indices = torch.stack([indices[1], indices[0]], dim=0)
            t_sparse_tensor = torch.sparse_coo_tensor(
                t_indices, values, size=(size[1], size[0])
            )
            grad_dense_x = torch.sparse.mm(t_sparse_tensor, grad_output)

        else:
            grad_dense_x = None

        return grad_sparse, grad_dense_x
