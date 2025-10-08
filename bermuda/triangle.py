from __future__ import annotations

import datetime
import re
import warnings
from collections import abc, defaultdict
from functools import cached_property
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import toolz as tlz

import altair as alt

from .base import Cell, IncrementalCell, Metadata, common_metadata, metadata_diff
from .date_utils import dev_lag_months, eval_date_resolution, period_resolution
from .errors import DuplicateCellWarning, TriangleEmptyError, TriangleError

__all__ = [
    "Triangle",
    "TriangleSlice",
]


class Triangle(abc.Set):
    """The main triangle class.

    The Triangle class inherits from collections.abc.Set.
    It holds a sequence of Cell objects, which are automatically
    sorted on class instance creation.

    The Triangle class has a number of methods, properties
    and static methods for triangle manipulation, visualization,
    and I/O. Consult the documentation for more details.

    Attributes:
        cells: the sequence of Cell objects.
    """

    def __init__(self, cells: Sequence[Cell]) -> None:
        if any(not isinstance(cell, Cell) for cell in cells):
            raise TriangleError("Bermuda triangles can only hold `Cell`s")

        # check for cell type consistency
        if not (
            all(cell.__class__.__name__ == "Cell" for cell in list(cells))
            or all(cell.__class__.__name__ == "CumulativeCell" for cell in list(cells))
            or all(cell.__class__.__name__ == "IncrementalCell" for cell in list(cells))
        ):
            raise TriangleError("Triangle cells must have consistent type.")

        try:
            self._cells = sorted(list(cells))
        except Exception as e:
            raise TriangleError(
                "Triangle must be constructed with some sequence of Cells"
            ) from e

        # The cells are already sorted by metadata and coordinates, so checking
        # for duplicates is as easy as checking for consecutive duplicates
        for before, after in zip(self._cells[:-1], self._cells[1:]):
            if (
                before.metadata == after.metadata
                and before.coordinates == after.coordinates
            ):
                warnings.warn(
                    "This triangle has multiple cells with the same metadata and coordinates "
                    f"at period_start {before.period_start} "
                    f"and evaluation_date {before.evaluation_date}",
                    DuplicateCellWarning,
                )
                break
            if isinstance(self._cells[0], IncrementalCell):
                if (
                    before.metadata == after.metadata
                    and before.period == after.period
                    and before.evaluation_date == after.evaluation_date
                    and before.prev_evaluation_date != after.prev_evaluation_date
                ):
                    warnings.warn(
                        "This incremental triangle has multiple cells with the same metadata "
                        f"and coordinates at period_start {before.period_start} "
                        f"and evaluation_date {before.evaluation_date} "
                        "with different prev_evaluation_dates",
                        DuplicateCellWarning,
                    )
                    break

    def __hash__(self):
        return hash(tuple(self._cells))

    def __eq__(self, other) -> bool:
        return all([cell1 == cell2 for cell1, cell2 in zip(self.cells, other.cells)])

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells)

    def __contains__(self, cell):
        return cell in self._cells

    def _repr_html_(self):
        if len(self) == 0:
            table = [
                "<table>",
                "  <thead>",
                "    <tr><th> <h4><b>Empty Triangle</b></h4> </th></tr>",
                "  </thead>",
                "  <tbody>",
                "    <tr><th> Number of slices: </th><td> 0 </td></tr>",
                "    <tr><th> Number of cells: </th><td> 0 </td></tr>",
                "  </tbody>",
                "</table>",
            ]
            return "\n".join(table)

        metadata_rows = []
        empty_column = "<td>   </td>"
        for name, value in self.common_metadata.as_flat_dict().items():
            if value is not None:
                metadata_rows.append(
                    f"<tr>{empty_column}<th>{name}</th> <td> {value} </td></tr>"
                )

        experience_gap_rows = []
        if self.experience_gaps:
            experience_gap_rows.append(
                '<tr><th colspan=2 style="text-align:center"> Experience Gaps </th></tr>'
            )
            for gap in self.experience_gaps:
                experience_gap_rows.append(
                    f"<tr><td></td><td>{gap[0]}/{gap[1]}</td></tr>"
                )

        standard_field_rows = []
        sparse_field_rows = []
        for name, count in self.field_cell_counts.items():
            if count == len(self):
                standard_field_rows.append(f"<tr>{empty_column}<td>{name}</td></tr>")
            else:
                hit_rate = 100 * count / len(self)
                sparse_field_rows.append(
                    f"<tr>{empty_column}<td>{name} ({hit_rate:.1f}% coverage)</td></tr>"
                )
        if standard_field_rows:
            standard_field_rows.insert(
                0, '<tr><th colspan=2 style="text-align:center"> Fields: </th></tr>'
            )
        if sparse_field_rows:
            sparse_field_rows.insert(
                0,
                '<tr><th colspan=2 style="text-align:center"> Optional Fields: </th></tr>',
            )

        if not self.is_slicewise_disjoint:
            category = "Non-Disjoint"
        elif not self.is_semi_regular():
            category = "Slicewise Disjoint"
        elif not self.is_regular():
            category = "Semi-Regular"
        else:
            category = "Regular"

        experience_range = f"{self.periods[0][0]}/{self.periods[-1][1]}"
        eval_date_range = f"{self.evaluation_dates[0]}/{self.evaluation_dates[-1]}"
        dev_lag_range = f"{self.dev_lags()[0]} - {self.dev_lags()[-1]}"
        tri_heading = (
            "Incremental Triangle"
            if isinstance(self._cells[0], IncrementalCell)
            else "Cumulative Triangle"
        )

        table = [
            f"<table>  <thead>    <tr><th> <h4><b>{tri_heading}</b></h4> </th></tr>",
            "  </thead>",
            "  <tbody>",
            f"    <tr><th> Number of slices: </th><td> {len(self.slices)} </td></tr>",
            f"    <tr><th> Number of cells: </th><td> {len(self)} </td></tr>",
            f"    <tr><th> Triangle category: </th><td> {category} </td></tr>",
            f"    <tr><th> Experience range: </th><td> {experience_range} </td></tr>",
            f"    <tr><th> Experience resolution: </th><td> {self.period_resolution} </td></tr>",
            f"    <tr><th> Evaluation range: </th><td> {eval_date_range} </td></tr>",
            f"    <tr><th> Evaluation resolution: </th><td> {self.eval_date_resolution} </td></tr>",
            f"    <tr><th> Dev Lag range: </th><td> {dev_lag_range} months </td></tr>",
            *experience_gap_rows,
            *standard_field_rows,
            *sparse_field_rows,
            '    <tr><th colspan=2 style="text-align:center"> Common Metadata: </th></tr>',
            *metadata_rows,
            "  </tbody>",
            "</table>",
        ]
        return "\n".join(table)

    def __repr__(self):
        return strip_html_tags(
            "\n".join(line.strip() for line in self._repr_html_().splitlines())
        )

    def __getitem__(self, index):
        """Triangle allows indexing by integer to directly retrieve a cell, or by
        (period, evaluation_date, metadata) indices. Period is indexed by period
        start. All indices can be passed as slices. Use `:` to indicate an empty
        slice selecting over all available cells. If a slice is passed to any index
        a Triangle object will be returned. If no slices are passed then the single
        Cell at the index will be returned."""
        # Direct Cell Indexing
        if isinstance(index, int):
            return self._cells[index]
        if isinstance(index, slice):
            return Triangle(self._cells[index])

        # Index by period, evaluation_date, metadata
        if len(index) == 3:
            period_slice, evaluation_slice, metadata = index
        else:
            raise ValueError("Must pass three indices to Triangle, or a single integer")

        if metadata and metadata != slice(None, None, None):
            filtered = self.filter(lambda cell: cell.metadata == metadata)
        else:
            filtered = self

        if isinstance(period_slice, slice):
            period_start = period_slice.start
            period_end = period_slice.stop
        elif isinstance(period_slice, datetime.date):
            period_start, period_end = period_slice, period_slice
        else:
            raise ValueError(
                f"period_slice must be date or slice, received {period_slice}"
            )

        # Clip cuts off based on period_end, we use filter to keep based on period_start
        if not period_start:
            period_start = datetime.date.min
        if not period_end:
            period_end = datetime.date.max
        filtered = filtered.filter(
            lambda cell: period_start <= cell.period_start <= period_end
        )

        if isinstance(evaluation_slice, slice):
            evaluation_start = evaluation_slice.start
            evaluation_end = evaluation_slice.stop
        elif isinstance(evaluation_slice, datetime.date):
            evaluation_start, evaluation_end = evaluation_slice, evaluation_slice
        else:
            raise ValueError(
                f"evaluation_slice must be date, received {evaluation_slice}"
            )

        clipped = filtered.clip(
            min_eval=evaluation_start,
            max_eval=evaluation_end,
        )
        if any(isinstance(ind, slice) for ind in index):
            return clipped
        return clipped._cells[0]

    @property
    def cells(self):
        return self._cells

    @property
    def slices(self) -> dict[Metadata, "Triangle"]:
        return tlz.valmap(
            Triangle, tlz.groupby(lambda cell: cell.metadata, self._cells)
        )

    @cached_property
    def metadata(self) -> list[Metadata]:
        """A sorted list of all unique metadata in the Triangle."""
        return sorted(set([cell.metadata for cell in self._cells]))

    @cached_property
    def eval_date_resolution(self) -> int:
        """The resolution of the evaluation dates in the triangle."""
        return eval_date_resolution(self)

    @cached_property
    def period_resolution(self) -> int:
        """The resolution of the periods in the triangle."""
        return period_resolution(self)

    @cached_property
    def common_metadata(self) -> Metadata:
        """A single metadata element representing the metadata attributes common to all
        Cells in the Triangle."""
        metas = self.metadata
        if len(metas) == 1:
            return metas[0]

        common_meta = metas[0]
        for meta in metas[1:]:
            common_meta = common_metadata(common_meta, meta)
        return common_meta

    @cached_property
    def metadata_differences(self) -> list[Metadata]:
        """A list of all unique metadata in the Triangle, with metadata attributes
        that are common to all Cells in the Triangle removed."""
        return [metadata_diff(self.common_metadata, meta) for meta in self.metadata]

    @cached_property
    def num_samples(self) -> int:
        """The number of samples in a Triangle of stochastic predictions. If the triangle
        has only observed data, `num_samples` is 1. If different cells or different fields
        have different values of `num_samples`, an error is raised."""
        num_samples = None
        for cell in self._cells:
            for val in cell.values.values():
                # Figure out the size of the value
                if isinstance(val, np.ndarray) and val.size > 1:
                    value_size = val.size
                else:
                    value_size = None

                # Compare and update the overall sample size
                if value_size is not None:
                    if num_samples is None:
                        num_samples = value_size
                    elif num_samples != value_size:
                        raise ValueError("The triangle has inconsistent value sizes")

        return 1 if num_samples is None else num_samples

    @property
    def period_rows(self):
        """Group by experience period and return an iterator over 'rows'."""
        grouped = tlz.groupby(lambda cell: cell.period, self._cells)
        for period in self.periods:
            yield (
                period,
                sorted(
                    grouped[period],
                    key=lambda cell: (cell.metadata, cell.evaluation_date),
                ),
            )

    @property
    def slice_period_rows(self):
        """Group by slice and experience period and return an iterator over 'slice-rows'."""
        grouped = tlz.groupby(lambda cell: (cell.metadata, cell.period), self._cells)
        for key, row in sorted(grouped.items()):
            yield key, sorted(row, key=lambda cell: cell.evaluation_date)

    @cached_property
    def periods(self) -> list[tuple[datetime.date, datetime.date]]:
        """A sorted list of unique experience periods (as 2-tuples of start and end dates)."""
        return sorted({cell.period for cell in self._cells})

    @cached_property
    def experience_gaps(self) -> list[tuple[datetime.date, datetime.date]]:
        """A sorted list of missing experience period ranges from the triangle"""
        missing_periods = []
        for (_, current_end), (next_start, _) in zip(
            self.periods[:-1], self.periods[1:]
        ):
            continuous_next_start = current_end + datetime.timedelta(days=1)
            if next_start != continuous_next_start:
                missing_periods += [
                    (continuous_next_start, next_start - datetime.timedelta(days=1))
                ]

        return missing_periods

    def dev_lags(self, unit="month") -> list[float | int | datetime.timedelta]:
        """A sorted list of unique development lags.  unit can be, 'month', 'day' or 'timedelta'.
        See docs for `Observation.dev_lag()` or `Prediction.dev_lag()` for more information.
        """
        return sorted({cell.dev_lag(unit) for cell in self._cells})

    @cached_property
    def evaluation_dates(self) -> list[datetime.date]:
        """A sorted list of unique evaluation dates."""
        return sorted({cell.evaluation_date for cell in self._cells})

    @cached_property
    def evaluation_date(self) -> datetime.date:
        """The latest evaluation date in the triangle."""
        if self.is_empty:
            raise TriangleEmptyError
        else:
            return max(self.evaluation_dates)

    @cached_property
    def fields(self) -> list[str]:
        """A sorted list of unique fields in the triangle set."""
        return sorted({name for cell in self._cells for name in cell.values.keys()})

    @cached_property
    def field_cell_counts(self) -> dict[str, int]:
        """For each triangle field return the number of cells the field appears in."""
        fcc = dict()
        for field in self.fields:
            fcc[field] = sum([field in cell.values.keys() for cell in self.cells])
        return fcc

    @cached_property
    def field_slice_counts(self) -> dict[str, int]:
        """For each triangle field return the number of slices the field appears in."""
        fsc = dict()
        for field in self.fields:
            fsc[field] = sum([field in slc.fields for slc in self.slices.values()])
        return fsc

    def __add__(self, other: Triangle) -> Triangle:
        """Concatenate `self` with `other` (another `Triangle`)."""
        return Triangle(self._cells + other._cells)

    def __radd__(self, other: Triangle) -> Triangle:
        # Enable summing triangle sets using built-in `sum` function.
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @cached_property
    def is_empty(self) -> bool:
        """Is the triangle empty?"""
        return self._cells == []

    @cached_property
    def is_disjoint(self) -> bool:
        """Is the triangle disjoint?

        A triangle is disjoint if there is no overlap between any pair of experience periods.
        """
        if self.is_empty:
            return True

        for (prev_start, prev_end), (next_start, next_end) in zip(
            self.periods[:-1], self.periods[1:]
        ):
            if prev_end >= next_start:
                return False
        return True

    @cached_property
    def is_slicewise_disjoint(self) -> bool:
        """Returns True if each triangle slice is disjoint."""
        for slc in self.slices.values():
            if not slc.is_disjoint:
                return False
        return True

    def is_semi_regular(self, dev_lag_unit="month") -> bool:
        """Is the triangle semi-regular?

        A triangle is semi-regular if it is disjoint and every experience period has the
        same duration."""
        if not self.is_disjoint:
            return False

        if self.is_empty:
            return True

        base_start, base_stop = self.periods[0]

        _unit = dev_lag_unit.lower()
        if "month" in _unit:
            # noinspection PyShadowingNames
            def diff_fn(start, stop):
                return dev_lag_months(start - datetime.timedelta(days=1), stop)

        elif "day" in _unit or _unit == "timedelta":
            # noinspection PyShadowingNames
            def diff_fn(start, stop):
                return start - stop

        else:
            raise ValueError(
                f"Unrecognized dev_lag unit '{dev_lag_unit}', "
                "must be one of 'month', 'day' or 'timedelta'."
            )

        base_length = diff_fn(base_start, base_stop)

        for start, stop in self.periods[1:]:
            if diff_fn(start, stop) != base_length:
                return False

        return True

    def is_regular(self, dev_lag_unit="month") -> bool:
        """Is the triangle regular?

        A triangle is regular if it is semi-regular and the interval between consecutive
        development lags is always a constant."""
        if not self.is_semi_regular(dev_lag_unit):
            return False
        if self.is_empty:
            return True

        dev_lags = self.dev_lags(unit=dev_lag_unit)
        # If there's only one dev lag, no need to check for consistent intervals
        if len(dev_lags) == 1:
            return True

        dev_offset = dev_lags[1] - dev_lags[0]
        for prv, nxt in zip(dev_lags[1:-1], dev_lags[2:]):
            if nxt - prv != dev_offset:
                return False

        return True

    @cached_property
    def has_consistent_currency(self) -> bool:
        """Does every cell in the triangle use the same currency?"""
        return len({cell.metadata.currency for cell in self.cells}) == 1

    @cached_property
    def has_consistent_risk_basis(self) -> bool:
        """Does every cell in the triangle use the same risk basis?"""
        # Risk basis is the third field in the metadata tuple
        return len({cell.metadata.risk_basis for cell in self.cells}) == 1

    @cached_property
    def is_incremental(self) -> bool:
        """Is the traingle incremental?"""
        return isinstance(self.cells[0], IncrementalCell)

    @cached_property
    def is_multi_slice(self) -> bool:
        """Does this triangle have multiple slices?"""
        return len(self.slices) > 1

    def derive_fields(self, **definitions: Callable[[Cell], Any] | Any) -> Triangle:
        """Create a new `Triangle` with additional derived fields defined.

        Args:
            **definitions: New field definitions. Argument names are the names of the
                new fields to be defined, values are either constants or functions that
                accept cells and return field values. Later definitions can depend on
                quantities computed via earlier definitions.
        """
        return Triangle(
            list(map(lambda cell: cell.derive_fields(**definitions), self.cells))
        )

    def derive_metadata(self, **definitions: Callable[[Cell], Any] | Any) -> Triangle:
        """Create a new `Triangle` with additional metadata attributes or fields derived.

        Args:
            **definitions: Values are plain values or functions that accept a `Cell` and
                    return a value. If keys are top-level attributes in metadata, their
                    values will replace the existing values of the top-level attributes.
                    Otherwise, the key-value pair will be added to the `details` field of
                    the metadata.
        """
        return Triangle(
            list(map(lambda cell: cell.derive_metadata(**definitions), self.cells))
        )

    def replace(self, **definitions: Callable[[Cell], Any] | Any) -> Triangle:
        """Replace cell member variables with a constant or via a function.

        Args:
            **definitions: New variable definitions, which must be one of
                `period_start`, `period_end`, `evaluation_date` or `values`.
                Definition values can either be constants or functions of the cell itself.
                For example,
                    triangle.replace(values=dict(reported_loss=1)
                    triangle.replace(values=lambda cell: dict(reported_loss = cell["paid_loss"]))

        """
        return Triangle(list(map(lambda cell: cell.replace(**definitions), self.cells)))

    def remove_static_details(self) -> Triangle:
        """Removes any metadata details and loss_details that are common to all cells in the triangle."""
        if self.is_empty:
            return self
        common_details = self.common_metadata.details.keys()
        common_loss_details = self.common_metadata.loss_details.keys()
        return self.derive_metadata(
            details=lambda meta: {
                k: v for k, v in meta.details.items() if k not in common_details
            },
            loss_details=lambda meta: {
                k: v
                for k, v in meta.loss_details.items()
                if k not in common_loss_details
            },
        )

    @cached_property
    def has_consistent_values_shapes(self) -> bool:
        """Does every field have the same shape in each cell?"""
        shapes = defaultdict(list)
        for cell in self.cells:
            for field, value in cell.values.items():
                shapes[field].append(np.size(value))

        return all(len(set(v)) == 1 for v in shapes.values())

    def select(self, keys: list[str]) -> Triangle:
        """Retain only the designated keys in `values`."""
        return Triangle(list(map(lambda cell: cell.select(keys), self.cells)))

    def clip(
        self,
        *,
        min_eval: datetime.date | None = None,
        max_eval: datetime.date | None = None,
        min_period: datetime.date | None = None,
        max_period: datetime.date | None = None,
        min_dev: datetime.timedelta | int | float | None = None,
        max_dev: datetime.timedelta | int | float | None = None,
        dev_lag_unit: str = "month",
    ) -> Triangle:
        """Create a new `Triangle` that has been clipped/censored as specified.

        Args:
            min_eval: Minimum evaluation date to permit.
            max_eval: Maximum evaluation date to permit.
            min_period: Minimum period start date to permit.
            max_period: Maximum period end date to permit.
            min_dev: Minimum development lag to permit.
            max_dev: Maximum development lag to permit.
            dev_lag_unit: Unit of min_dev or max_dev, required if either is specified.
        """
        cells = self._cells
        if min_eval is not None:
            cells = filter(lambda cell: cell.evaluation_date >= min_eval, cells)
        if max_eval is not None:
            cells = filter(lambda cell: cell.evaluation_date <= max_eval, cells)
        if min_period is not None:
            cells = filter(lambda cell: cell.period_start >= min_period, cells)
        if max_period is not None:
            cells = filter(lambda cell: cell.period_end <= max_period, cells)
        if min_dev is not None:
            cells = filter(lambda cell: cell.dev_lag(dev_lag_unit) >= min_dev, cells)
        if max_dev is not None:
            cells = filter(lambda cell: cell.dev_lag(dev_lag_unit) <= max_dev, cells)

        return Triangle(list(cells))

    def filter(self, predicate: Callable[[Cell], bool]) -> Triangle:
        """Create a new `Triangle` with only the cells that satisfy the predicate.

        Args:
            predicate: A function that accepts a `Cell` and returns a `boolean`.
        """
        return Triangle(list(filter(predicate, self._cells)))

    def extract(self, attribute: Callable[[Cell], Any] | str) -> np.ndarray:
        """Extract cell attributes either via a function or cell field name,
        returning a numpy array of values.

        Args:
            attribute: A function applied to all cells or a cell field.
        """
        if isinstance(attribute, Callable):
            return np.array(list(map(attribute, self._cells)))
        return np.array(list(map(lambda cell: cell.values.get(attribute), self._cells)), dtype="object")

    def to_data_frame(self, dev_lag_unit="month"):
        """Convert a triangle set to a Pandas dataframe."""
        return pd.DataFrame(
            cell.to_record(dev_lag_unit) for cell in self._cells
        ).reset_index(drop=True)

    @property
    def right_edge(self):
        """Create a `Triangle` with only the most recent observation for each period
        within each triangle slice.

        Note that right_edge doesn't guarantee that every cell will have the same evaluation date!
        """
        new_cells = []
        for slc in self.slices.values():
            for period, row in slc.period_rows:
                new_cells.append(row[-1])
        return Triangle(new_cells)

    @property
    def is_right_edge_ragged(self) -> bool:
        """Returns `True` if any slice's right edge encompasses different evaluation dates"""
        for slc in self.slices.values():
            if len(slc.right_edge.evaluation_dates) > 1:
                return True
        return False


class TriangleSlice(Triangle):
    def __init__(self, cells: Sequence[Cell]):
        super().__init__(cells)
        if len(self.slices) > 1:
            raise TriangleError("TriangleSlice cannot have multiple slices.")

    def __getitem__(self, index):
        """TriangleSlice allows indexing by integer to directly retrieve a cell, or by
        (period, evaluation_date) indices. Period is indexed by period
        start. All indices can be passed as slices. Use `:` to indicate an empty
        slice selecting over all available cells. If a slice is passed to any index
        a TriangleSlice object will be returned. If no slices are passed then the single
        Cell at the index will be returned."""
        # Direct Cell Indexing
        if isinstance(index, int):
            return self._cells[index]
        if isinstance(index, slice):
            return TriangleSlice(self._cells[index])

        # Index by period, evaluation_date
        if len(index) == 2:
            period_slice, evaluation_slice = index
        else:
            raise ValueError(
                "Must pass two indices to TriangleSlice, or a single integer"
            )

        if isinstance(period_slice, slice):
            period_start = period_slice.start
            period_end = period_slice.stop
        elif isinstance(period_slice, datetime.date):
            period_start, period_end = period_slice, period_slice
        else:
            raise ValueError(
                f"period_slice must be date or slice, received {period_slice}"
            )

        # Clip cuts off based on period_end, we use filter to keep based on period_start
        if not period_start:
            period_start = datetime.date.min
        if not period_end:
            period_end = datetime.date.max

        filtered = self.filter(
            lambda cell: period_start <= cell.period_start <= period_end
        )

        if isinstance(evaluation_slice, slice):
            evaluation_start = evaluation_slice.start
            evaluation_end = evaluation_slice.stop
        elif isinstance(evaluation_slice, datetime.date):
            evaluation_start, evaluation_end = evaluation_slice, evaluation_slice
        else:
            raise ValueError(
                f"evaluation_slice must be date , received {evaluation_slice}"
            )

        clipped = filtered.clip(
            min_eval=evaluation_start,
            max_eval=evaluation_end,
        )
        if any(isinstance(ind, slice) for ind in index):
            return TriangleSlice(clipped.cells)
        return clipped._cells[0]


def strip_html_tags(data):
    p = re.compile(r"<.*?>")
    return p.sub("", data)
