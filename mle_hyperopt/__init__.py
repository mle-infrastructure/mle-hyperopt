from ._version import __version__
from .strategies import (
    RandomSearch,
    GridSearch,
    SMBOSearch,
    NevergradSearch,
    CoordinateSearch,
    HalvingSearch,
    HyperbandSearch,
    PBTSearch,
)
from .decorator import hyperopt

Strategies = {
    "Random": RandomSearch,
    "Grid": GridSearch,
    "SMBO": SMBOSearch,
    "Nevergrad": NevergradSearch,
    "Coordinate": CoordinateSearch,
    "Halving": HalvingSearch,
    "Hyperband": HyperbandSearch,
    "PBT": PBTSearch,
}

__all__ = [
    "__version__",
    "RandomSearch",
    "GridSearch",
    "SMBOSearch",
    "NevergradSearch",
    "CoordinateSearch",
    "HalvingSearch",
    "HyperbandSearch",
    "PBTSearch",
    "hyperopt",
    "Strategies",
]
