from .__about__ import __version__
import os

from .base import Cell, CumulativeCell, IncrementalCell, Metadata

from .errors import *
from .io import *
from .matrix import (
    DisaggregatedValue,
    Matrix,
    MatrixIndex,
    MissingValue,
    PredictedValue,
    RichMatrix,
)

from .utils import *
from .factory import Triangle
from .triangle import TriangleSlice
from .date_utils import *

local_dir = os.path.dirname(os.path.abspath(__file__))
meyers_tri = Triangle.from_binary(os.path.join(local_dir, "meyers.trib"))
