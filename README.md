## phlower_tensor


### Description

This library is separated from [the main phlower library](https://github.com/ricosjp/phlower) to provide tensor operations and utilities that can be used independently. It includes tensor objects which have physical dimensions, as well as various tensor operations such as addition, multiplication, contraction, and more.



### Installation

You can install the library using pip:

```bash
pip install phlower-tensor
```


### Usage

Here is a simple example of how to use the phlower_tensor library:

```python
import torch
from phlower_tensor import phlower_tensor

# Create two tensors with physical dimensions
velocity = phlower_tensor(torch.rand(100, 3), dimension={"T": -1, "L": 1})
mass = phlower_tensor(torch.rand(100, 1), dimension={"M": 1})

kinetic_energy = 0.5 * mass * (velocity ** 2)
print(kinetic_energy.dimension)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)

```


Physical dimensions are represented as dictionaries where keys are dimension symbols (e.g., "L" for length, "T" for time) and values are their respective exponents.

| Symbol | Dimension      |
|--------|----------------|
| L      | Length         |
| T      | Time           |
| M      | Mass           |
| I      | Electric Current |
| Theta  | Temperature    |
| N      | Amount of Substance |
| J      | Luminous Intensity |


### License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

