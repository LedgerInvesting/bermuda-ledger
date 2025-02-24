from dataclasses import dataclass, field

import numpy as np

from .index import MatrixIndex

__all__ = [
    "MissingValue",
    "DisaggregatedValue",
    "PredictedValue",
    "DisaggregatedPredictedValue",
    "Matrix",
    "RichMatrix",
]


@dataclass
class MissingValue(object):
    id: int


@dataclass(frozen=True)
class DisaggregatedValue(object):
    id: int
    value: float


@dataclass(frozen=True)
class PredictedValue(object):
    value: np.ndarray | None = field(default=None)


@dataclass(frozen=True)
class DisaggregatedPredictedValue(object):
    id: int
    value: np.ndarray | None = field(default=None)


class Matrix(object):
    def __init__(self, data: np.ndarray, index: MatrixIndex, incremental: bool = False):
        self.data = data
        self.index = index
        self.incremental = incremental

    def subset(self, args) -> "Matrix":
        int_slices, int_fields, int_exps, int_devs = self.index.resolve_indices(*args)
        return Matrix(
            data=self.data[int_slices, int_fields, int_exps, int_devs],
            index=self.index.subset(
                self.data.shape, int_slices, int_fields, int_exps, int_devs
            ),
        )

    def __getitem__(self, args) -> np.ndarray:
        int_slices, int_fields, int_exps, int_devs = self.index.resolve_indices(*args)
        return self.data[int_slices, int_fields, int_exps, int_devs]

    def __setitem__(self, args, value):
        int_slices, int_fields, int_exps, int_devs = self.index.resolve_indices(*args)
        self.data[int_slices, int_fields, int_exps, int_devs] = value


class RichMatrix(Matrix):
    pass
