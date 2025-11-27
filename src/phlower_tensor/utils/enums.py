from enum import Enum


class PhysicalDimensionSymbolType(Enum):
    T = 0  # time
    L = 1  # length
    M = 2  # mass
    I = 3  # electric current  # noqa: E741
    Theta = 4  # thermodynamic temperature
    N = 5  # amount of substance
    J = 6  # luminous intensity

    @classmethod
    def is_exist(cls, name: str) -> bool:
        return name in _symbol2quntityname.keys()

    def to_quantity_name(self) -> str:
        return _symbol2quntityname[self.name]

    def __str__(self):
        return f"{self.name} ({self.to_quantity_name()})"


_symbol2quntityname = {
    PhysicalDimensionSymbolType.T.name: "time",
    PhysicalDimensionSymbolType.L.name: "length",
    PhysicalDimensionSymbolType.M.name: "mass",
    PhysicalDimensionSymbolType.I.name: "electric current",
    PhysicalDimensionSymbolType.Theta.name: "thermodynamic temperature",
    PhysicalDimensionSymbolType.N.name: "amount of substance",
    PhysicalDimensionSymbolType.J.name: "luminous intensity",
}
