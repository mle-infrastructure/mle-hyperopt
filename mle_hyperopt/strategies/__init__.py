from .random import RandomSearch
from .grid import GridSearch
from .smbo import SMBOSearch
from .nevergrad import NevergradSearch
from .coordinate import CoordinateSearch
from .halving import HalvingSearch
from .hyperband import HyperbandSearch
from .pbt import PBTSearch

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
    "Strategies",
]
