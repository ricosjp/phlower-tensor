from __future__ import annotations

from phlower_tensor.utils.enums import PhysicalDimensionSymbolType
from phlower_tensor.utils.exceptions import InvalidDimensionError


class PhysicalDimensions:
    def __init__(self, dimensions: dict[str, float]) -> None:
        self._check(dimensions)

        self._dimensions = dict.fromkeys(
            PhysicalDimensionSymbolType.__members__, 0.0
        )
        self._dimensions |= dimensions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhysicalDimensions):
            return False

        for k in PhysicalDimensionSymbolType.__members__.keys():
            if self._dimensions[k] != other[k]:
                return False

        return True

    def _check(self, dimensions: dict[str, float]):
        for name in dimensions:
            if not PhysicalDimensionSymbolType.is_exist(name):
                raise InvalidDimensionError(
                    f"dimension name: {name} is not implemented."
                    f"Avaliable units are {list(PhysicalDimensionSymbolType)}"
                )
            if dimensions[name] is None:
                raise InvalidDimensionError(
                    f"None dimension is found in {name}"
                )

    def get(self, name: str) -> float | None:
        return self._dimensions.get(name)

    def __getitem__(self, name: str) -> float:
        if not PhysicalDimensionSymbolType.is_exist(name):
            raise InvalidDimensionError(f"{name} is not valid dimension name.")
        return self._dimensions[name]

    def to_list(self) -> list[float]:
        _list: list[float] = [0 for _ in PhysicalDimensionSymbolType]
        for k, v in self._dimensions.items():
            if k not in PhysicalDimensionSymbolType.__members__:
                raise InvalidDimensionError(
                    f"dimension name: {k} is not implemented."
                    f"Avaliable units are {list(PhysicalDimensionSymbolType)}"
                )
            _list[PhysicalDimensionSymbolType[k].value] = v

        return _list

    def to_dict(self) -> dict:
        return self._dimensions
