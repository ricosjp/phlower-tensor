class InvalidDimensionError(ValueError):
    """This error raises when invalid dimension is found."""


class DimensionIncompatibleError(ValueError):
    """This error raises when calculation of physical units failed due to
    unit incompatibility.
    """


class PhlowerSparseUnsupportedError(ValueError):
    """
    This error raises when trying to call methods not supported for sparse
    tensors
    """


class PhlowerUnsupportedTorchFunctionError(ValueError):
    """
    This error raises when trying to call a function not supported
    by the phlower library although torch does
    """


class PhlowerIncompatibleTensorError(ValueError):
    """
    This error raises when trying to perform an operation for incompatible
    tensor(s)
    """
