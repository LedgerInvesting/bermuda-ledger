import datetime
from warnings import warn

import numpy as np

from ..base import Metadata
from ..date_utils import eval_date_resolution, month_to_id, period_resolution
from ..triangle import Triangle

__all__ = [
    "MatrixIndex",
]


class MatrixIndex(object):
    """Utility class to handle natural indexing of Matrix-style Bermuda data."""

    def __init__(
        self,
        slices: list[Metadata],
        fields: list[str],
        exp_origin: int,
        dev_origin: int,
        exp_resolution: int,
        dev_resolution: int,
    ):
        self._slices = slices
        self._slice_lookup = {meta: ndx for ndx, meta in enumerate(self._slices)}
        self._fields = fields
        self._field_lookup = {field: ndx for ndx, field in enumerate(self._fields)}
        self._exp_origin = exp_origin
        self._exp_resolution = exp_resolution
        self._dev_origin = dev_origin
        self._dev_resolution = dev_resolution

    slices = property(lambda self: self._slices)
    slice_lookup = property(lambda self: self._slice_lookup)
    fields = property(lambda self: self._fields)
    field_lookup = property(lambda self: self._field_lookup)
    exp_origin = property(lambda self: self._exp_origin)
    exp_resolution = property(lambda self: self._exp_resolution)
    dev_origin = property(lambda self: self._dev_origin)
    dev_resolution = property(lambda self: self._dev_resolution)

    @classmethod
    def from_triangle(
        cls,
        tri: Triangle,
        eval_resolution: int | None = None,
        fields: list[str] | None = None,
    ) -> "MatrixIndex":
        # Get experience indices
        exp_origin = month_to_id(min([start for start, _ in tri.periods]))
        exp_resolution = period_resolution(tri)
        # Get dev lag indices
        int_dev_lags = [int(lag) for lag in tri.dev_lags()]
        dev_origin = min(int_dev_lags)
        dev_resolution = eval_date_resolution(tri)
        if not fields:
            fields = tri.fields
            if not fields:
                raise Exception("Must provide fields if Triangle has none")

        if not eval_resolution:
            if not dev_resolution:
                raise Exception(
                    "Must supply eval_resolution for single dev lag Triangle"
                )
            eval_resolution = dev_resolution
        else:
            if dev_resolution and eval_resolution > dev_resolution:
                warn(
                    f"Lowering resolution from {dev_resolution} months to {eval_resolution} months"
                )

        return cls(
            slices=tri.metadata,
            fields=fields,
            exp_origin=exp_origin,
            exp_resolution=exp_resolution,
            dev_origin=dev_origin,
            dev_resolution=eval_resolution,
        )

    def subset(self, shape, *args) -> "MatrixIndex":
        if len(args) != 4:
            raise Exception("Subsetting expects exactly 4 arguments")
        slice_ndxs, field_ndxs, exp_ndxs, dev_ndxs = args
        slice_ndxs = [slice_ndxs] if isinstance(slice_ndxs, int) else slice_ndxs
        field_ndxs = [field_ndxs] if isinstance(field_ndxs, int) else field_ndxs

        if isinstance(exp_ndxs, slice):
            first_exp = 0 if exp_ndxs.start is None else exp_ndxs.start
        else:
            first_exp = exp_ndxs
        if isinstance(dev_ndxs, slice):
            first_dev = 0 if dev_ndxs.start is None else dev_ndxs.start
        else:
            first_dev = dev_ndxs

        _, _, exp_size, dev_size = shape
        if first_exp < 0:
            first_exp = exp_size - first_exp
        if first_dev < 0:
            first_dev = dev_size - first_dev

        return MatrixIndex(
            slices=[self._slices[ndx] for ndx in slice_ndxs],
            fields=[self._fields[ndx] for ndx in field_ndxs],
            exp_origin=self._exp_origin + self._exp_resolution * first_exp,
            exp_resolution=self._exp_resolution,
            dev_origin=self._dev_origin + self._dev_resolution * first_dev,
            dev_resolution=self._dev_resolution,
        )

    def resolve_indices(self, *args):
        if len(args) != 4:
            raise Exception("Matrix indexing expects exactly 4 arguments")
        slice_, field, exp, dev = args
        return (
            self._resolve_index(slice_, self._resolve_slice_ndx),
            self._resolve_index(field, self._resolve_field_ndx),
            self._resolve_index(exp, self._resolve_exp_ndx, False),
            self._resolve_index(dev, self._resolve_dev_ndx, False),
        )

    @staticmethod
    def _resolve_index(arg, ndx_fn, allow_iterable=True):
        if isinstance(arg, (list, np.ndarray)):
            if allow_iterable:
                return [ndx_fn(elem) for elem in arg]
            else:
                raise Exception(
                    "Indexing by an iterable on this dimension is not allowed"
                )
        elif isinstance(arg, slice):
            return slice(ndx_fn(arg.start), ndx_fn(arg.stop), arg.step)
        else:
            return ndx_fn(arg)

    def _resolve_slice_ndx(self, arg):
        if arg is None:
            return None
        if isinstance(arg, int):
            if arg < 0:
                raise Exception(f"Slice index {arg} is out of range")
            return arg
        elif isinstance(arg, Metadata):
            return self._slice_lookup[arg]
        else:
            raise Exception(f"Unknown slice indexing type {arg.__class__.__name__}")

    def _resolve_field_ndx(self, arg):
        if arg is None:
            return None
        elif isinstance(arg, int):
            if arg < 0:
                raise Exception(f"Field index {arg} is out of range")
            return arg
        elif isinstance(arg, str):
            return self._field_lookup[arg]
        else:
            raise Exception(f"Unknown field indexing type {arg.__class__.__name__}")

    def _resolve_exp_ndx(self, arg):
        if arg is None:
            return None
        elif isinstance(arg, int):
            if arg < 0:
                raise Exception(f"Experience index {arg} is out of range")
            return arg
        elif isinstance(arg, datetime.date):
            arg_id = month_to_id(arg)
            arg_ndx = (arg_id - self._exp_origin) // self._exp_resolution
            if arg_ndx < 0:
                raise Exception(f"Experience date {arg} is out of range")
            return arg_ndx
        else:
            raise Exception(
                f"Unknown experience period indexing type {arg.__class__.__name__}"
            )

    def _resolve_dev_ndx(self, arg):
        if arg is None:
            return None
        elif isinstance(arg, float):
            # Triangles may have unequal experience period and development lag
            # resolutions, so we need to choose the smallest resolution
            # to create the resolved index.
            ndx_resolution = min(self._dev_resolution, self._exp_resolution)
            arg_ndx = int((arg - self._dev_origin) / ndx_resolution)
            if arg_ndx < 0:
                raise Exception(f"Development lag {arg} is out of range")
            return arg_ndx
        elif isinstance(arg, int):
            if arg < 0:
                raise Exception(f"Development index {arg} is out of range")
            return arg
        else:
            raise Exception(f"Unknown dev lag indexing type {arg.__class__.__name__}")
