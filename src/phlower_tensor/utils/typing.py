import numpy as np
import scipy.sparse as sp

DenseArrayType = np.ndarray


# _sparse_array_wrapper.py uses `col` attribute in `to_tensor` method
# When you want to activate csr and csc matrix, modify them.
SparseArrayType = (
    sp.coo_matrix
    # | sp.csr_matrix
    # | sp.csc_matrix
    # | sp.csr_array
    | sp.coo_array
    # | sp.csc_array
)

ArrayDataType = np.ndarray | SparseArrayType
