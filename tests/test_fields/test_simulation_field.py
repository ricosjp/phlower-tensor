import pytest
import torch

from phlower_tensor import SimulationField
from phlower_tensor.collections import phlower_tensor_collection


def test__overwrite_simulation_field():
    generator = torch.Generator()
    current_data = {
        "field1": torch.rand(10, 3, generator=generator),
        "field2": torch.rand(10, 5, generator=generator),
    }
    new_data = {
        "field1": torch.rand(10, 3, generator=generator),
        "field2": torch.rand(10, 5, generator=generator),
        "field3": torch.rand(10, 2, generator=generator),
    }

    for key in current_data.keys():
        assert not torch.allclose(current_data[key], new_data[key])

    simulation_field = SimulationField(phlower_tensor_collection(current_data))
    updated_field = simulation_field.overwrite(new_data)

    assert isinstance(updated_field, SimulationField)
    assert updated_field.keys() == new_data.keys()
    for key in updated_field.keys():
        assert torch.allclose(updated_field[key].to_tensor(), new_data[key])

    # check that the original simulation field is not modified
    for key in current_data.keys():
        assert torch.allclose(
            simulation_field[key].to_tensor(), current_data[key]
        )


def test__overwrite_simulation_field_with_inconsistent_shape():
    generator = torch.Generator()
    current_data = {
        "field1": torch.rand(10, 3, generator=generator),
        "field2": torch.rand(10, 5, generator=generator),
    }
    new_data = {
        "field1": torch.rand(10, 4, generator=generator),  # Inconsistent shape
        "field2": torch.rand(10, 5, generator=generator),
    }

    simulation_field = SimulationField(phlower_tensor_collection(current_data))

    with pytest.raises(ValueError):
        simulation_field.overwrite(new_data)
