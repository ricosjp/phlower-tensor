# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.3

### Added

* Add supported tensor operations:
    - `torch.roll`
    - `torch.median`
    - `torch.linalg.norm`
    - `torch.linalg.vector_norm`
    - `torch.clamp`
    - `torch.repeat`
    - `torch.index_add`
    - `torch.linalg.cholesky`
    - `torch.scatter_add` 
    - `torch.cos`
    - `torch.linalg.pinv`
    - `torch.index_put`
* Add `ndim` and `dim` properties to `PhlowerTensor` class
* Support `csr` and `csc` sparse tensor formats in `PhlowerTensor` class
* Add new concatenation mode index_shifting
* Add `overwrite` method to SimulationField and its interface
* Add `requires_grad_` method to IPhlowerTensorCollections


### Changed

* Improve `spmm` function to consume less memory by defining a custom backward function
* Add `auto` keyword to `einsum` function to automatically infer the output physical dimension


### Fixed

* Fix dtype when computing `torch.scatter` with `PhlowerTensor` inputs
* Fix to coalesce tensors when performing batch operation for non-coalesced tensors
