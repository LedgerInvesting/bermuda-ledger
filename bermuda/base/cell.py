from __future__ import annotations

import dataclasses
import datetime
from collections import namedtuple
from typing import Any, Callable, Tuple, Union, get_args

import numpy as np

from ..date_utils import calculate_dev_lag, month_to_id
from .metadata import METADATA_COMMON_ATTRIBUTES, Metadata

__all__ = [
    "CellValue",
    "Cell",
    "values_eq",
    "format_value",
]


_constr_coordinates = namedtuple(
    "coordinates", ["period_start", "period_end", "evaluation_date"]
)
CellValue = Union[float, int, np.ndarray, np.float64, np.int64, None]


class Cell(object):
    """Base class for all elements contained within `Triangle`s.

    Attributes:
        period_start: The beginning of the experience period.
        period_end: The end of the experience period (as a closed interval).
        evaluation_date: The date when the information was known.
        values: dictionary with key-value pairs of fields and values.
            For example: `{"paid_loss": 123400, "earned_premium": 2345000}`
        metadata: Other metadata about the `Cell`.
    """

    def __init__(
        self,
        period_start: datetime.date,
        period_end: datetime.date,
        evaluation_date: datetime.date,
        values: dict[str, CellValue],
        metadata: Metadata = None,
        _skip_validation: bool = False,
    ):
        if not _skip_validation:
            # Simple input type checks
            if not isinstance(period_start, datetime.date):
                raise TypeError("`period_start` must be a `datetime.date`")
            if not isinstance(period_end, datetime.date):
                raise TypeError("`period_end` must be a `datetime.date`")
            if not isinstance(evaluation_date, datetime.date):
                raise TypeError("`evaluation_date` must be a `datetime.date`")
            if not isinstance(metadata, (Metadata, type(None))):
                raise TypeError("`metadata` must be a `Metadata` object (or `None`)")

            # Check that all values are of the correct type (or can be coerced to the correct type)
            if not isinstance(values, dict):
                raise TypeError("`values` must be a `dict`")
            for name, val in values.items():
                if not isinstance(val, get_args(CellValue)):
                    raise TypeError(
                        f"Field {name} has unexpected value type {val.__class__.__name__}"
                    )
                if isinstance(val, np.ndarray):
                    # See binary_output.py line 212
                    if not isinstance(val.dtype, (np.int64, np.float64)):
                        try:
                            val = val.astype(np.float64)
                        except ValueError as e:
                            raise ValueError(
                                f"""
                            Error: The array in {name} is not of dtype `np.int64` or `np.float64`,
                            nor is it coercible to `np.float64`. Outputting to binary format is only
                            compatible with 64-bit numeric types.
                            """
                            ) from e

            # Check for date consistency
            if period_end < period_start:
                raise ValueError("`period_end` must be >= `period_start`")
            if evaluation_date < period_start:
                raise ValueError(
                    f"`evaluation_date` ({evaluation_date}) must be >= `period_start` ({period_start})."
                )
            if evaluation_date == datetime.date.max:
                raise ValueError(
                    f"`evaluation_date` ({evaluation_date}) must be < `datetime.date.max`."
                )

        # Store dates as datetime.date objects (dropping time component, if present)
        self._period_start = datetime.date(
            period_start.year, period_start.month, period_start.day
        )
        self._period_end = datetime.date(
            period_end.year, period_end.month, period_end.day
        )
        self._evaluation_date = datetime.date(
            evaluation_date.year, evaluation_date.month, evaluation_date.day
        )

        self._values = values if values is not None else {}
        self._metadata = metadata if metadata is not None else Metadata()

    # === Accessors =============================================================================
    # Shorthand implementation for properties on all basic fields
    period_start = property(lambda self: self._period_start)
    period_end = property(lambda self: self._period_end)
    evaluation_date = property(lambda self: self._evaluation_date)
    values = property(lambda self: self._values)
    metadata = property(lambda self: self._metadata)
    risk_basis = property(lambda self: self._metadata.risk_basis)
    country = property(lambda self: self._metadata.country)
    currency = property(lambda self: self._metadata.currency)
    reinsurance_basis = property(lambda self: self._metadata.reinsurance_basis)
    loss_definition = property(lambda self: self._metadata.loss_definition)
    per_occurrence_limit = property(lambda self: self._metadata.per_occurrence_limit)
    details = property(lambda self: self._metadata.details)
    loss_details = property(lambda self: self._metadata.loss_details)

    @property
    def period(self) -> Tuple[datetime.date, datetime.date]:
        return self._period_start, self._period_end

    @property
    def period_length(self) -> int:
        """The length of the cell experience period, in months."""
        return month_to_id(self._period_end) - month_to_id(self._period_start) + 1

    @property
    def coordinates(self) -> Tuple[datetime.date, datetime.date, datetime.date]:
        """Location of the cell: period_start, period_end, evaluation_date."""
        return _constr_coordinates(
            self._period_start, self._period_end, self._evaluation_date
        )

    def dev_lag(self, unit: str = "months") -> Union[float, int, datetime.timedelta]:
        """Compute the development lag of the cell in the specified units.

        Args:
            unit (str): One of 'month', 'day', 'timedelta'.  'month' returns dev_lag as
                a float number of months, day returns dev_lag as an integer number
                of days, and 'timedelta' returns a datetime.timedelta object.
        """
        return calculate_dev_lag(self._period_end, self._evaluation_date, unit)

    # === Dunder methods ========================================================================

    def __hash__(self):
        # Numpy arrays aren't hashable, so manually handle the values dictionary
        value_hashes = []
        for k, v in self._values.items():
            if isinstance(v, np.ndarray):
                item_hash = hash((k, tuple(v)))
            else:
                item_hash = hash((k, v))
            value_hashes.append(item_hash)

        return hash(
            (
                # Including class name to ensure cumulative and incremental are distinct
                self.__class__.__name__,
                self._period_start,
                self._period_end,
                self._evaluation_date,
                self._metadata,
                # Sorting value hashes here to ensure insensitivity to
                # key insertion order
                tuple(sorted(value_hashes)),
            )
        )

    def __getitem__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError:
            raise KeyError(
                f"Field `{key}` does not exist in this `{self.__class__.__name__}`"
            )

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __eq__(self, other: Cell) -> bool:
        return (
            # Defining class equality this way to handle cases where Cells are being
            # compared to CumulativeCells or vice-versa.
            (isinstance(self, other.__class__) or isinstance(other, self.__class__))
            and self._period_start == other._period_start
            and self._period_end == other._period_end
            and self._evaluation_date == other._evaluation_date
            and self._metadata == other._metadata
            and values_eq(self._values, other._values)
        )

    def __lt__(self, other: Cell) -> bool:
        """Required so that lists of cells can be sortable.
        Comparison is based on (period_start, period_end, evaluation_date) tuple.
        """
        return (
            self._metadata,
            self._period_start,
            self._period_end,
            self._evaluation_date,
        ) < (
            other._metadata,
            other._period_start,
            other._period_end,
            other._evaluation_date,
        )

    def __repr__(self) -> str:
        attr_strings = []
        for name, val in self.__dict__.items():
            attr_name = name[1:] if name[0] == "_" else name
            attr_strings.append(f"{attr_name}={repr(val)}")
        return self.__class__.__name__ + "(" + ", ".join(attr_strings) + ")"

    def _repr_html_(self) -> str:
        metadata_rows = []
        for name, value in self.metadata.as_flat_dict().items():
            if value is not None:
                metadata_rows.append(f"<tr><th> {name} </th> <td> {value} </td></tr>")

        value_rows = []
        for name, value in self.values.items():
            if isinstance(value, np.ndarray):
                # noinspection PyTypeChecker
                value_summary = f"~{format_value(np.mean(value))} (Size {value.size})"
            else:
                value_summary = format_value(value)
            value_rows.append(f"<tr><th> {name} </th> <td> {value_summary} </td></tr>")

        table = [
            "<table>",
            "  <thead>",
            f"    <tr><th> <h4><b>{self.__class__.__name__}</b></h4> </th></tr>",
            "  </thead>",
            "  <tbody>",
            f"    <tr><th> Period Start: </th><td> {self.period_start} </td></tr>",
            f"    <tr><th> Period End: </th><td> {self.period_end} </td></tr>",
            f"    <tr><th> Evaluation Date: </th><td> {self.evaluation_date} </td></tr>",
            f"    <tr><th> Development Lag: </th><td> {self.dev_lag()} months </td></tr>",
            '    <tr><th colspan=2 style="text-align:center"> Values </th></tr>',
            *value_rows,
            '    <tr><th colspan=2 style="text-align:center"> Metadata </th></tr>',
            *metadata_rows,
            "  </tbody></table>",
        ]
        return "\n".join(table)

    # === Transformer methods ===================================================================

    def _base_replace(self, **definitions) -> Cell:
        """Override values of arbitrary fields in a non-destructive fashion."""
        attrs = {k[1:]: v for k, v in self.__dict__.items()}
        attrs.update(definitions)
        return self.__class__(**attrs)

    def replace(self, **definitions: Union[Callable[[Cell], Any], Any]) -> Cell:
        """Override values of arbitary fields with constants or functions.
        Calls _base_replace internally. Only validates the final cell once
        all replacements have been made.
        """
        cell = self
        for name, func_or_val in definitions.items():
            value = func_or_val(cell) if callable(func_or_val) else func_or_val
            definitions = {name: value}
            cell = cell._base_replace(**definitions, _skip_validation=True)
        return cell._base_replace(_skip_validation=False)

    def select(self, keys: list[str]) -> Cell:
        """Retain only the designated keys in `values`."""
        new_values = {k: v for k, v in self.values.items() if k in keys}
        return self.replace(values=new_values)

    def derive_fields(self, **definitions: Union[Callable[[Cell], Any], Any]) -> Cell:
        """Add new derived fields to a cell.

        Arguments:
            **definitions: Values are functions that accept a `Cell` and return a value,
                and keys are the names that the function values will be assigned to.
        Examples:
            >>> ob = Cell(
            ...     period_start=0,
            ...     period_end=1,
            ...     values={
            ...         "loss": 10,
            ...         "premium": 50,
            ...     },
            ... )
            >>> new_ob = ob.derive_fields(
            ...     loss_ratio=lambda c: c[
            ...         "loss"
            ...     ]
            ...     / c["premium"]
            ... )
            >>> new_ob.values
            {"loss": 10, "premium": 50, "loss_ratio": 0.2}
        """
        cell = self
        for name, func_or_val in definitions.items():
            value = func_or_val(cell) if callable(func_or_val) else func_or_val
            definitions = {name: value}
            cell = cell.replace(values={**cell.values, **definitions})
        return cell

    def derive_metadata(self, **definitions: Union[Callable[[Cell], Any], Any]) -> Cell:
        """Add new derived attributes to a Cell's metadata.

        Arguments:
            **definitions: Values are plain values or functions that accept a `Cell` and
                return a value. If keys are top-level attributes in metadata, their
                values will replace the existing values of the top-level attributes.
                Otherwise, the key-value pair will be added to the `details` field of
                the metadata.
        """
        cell = self
        for name, func_or_val in definitions.items():
            value = func_or_val(cell) if callable(func_or_val) else func_or_val
            if name in METADATA_COMMON_ATTRIBUTES:
                new_metadata = dataclasses.replace(cell.metadata, **{name: value})
            else:
                new_details = {**cell.metadata.details, name: value}
                new_metadata = dataclasses.replace(cell.metadata, details=new_details)
            cell = cell._base_replace(metadata=new_metadata)
        return cell

    def to_record(self, dev_lag_unit="month") -> dict[str, Any]:
        """Return dict representation of Cell."""
        return {
            "period_start": self._period_start,
            "period_end": self._period_end,
            "evaluation_date": self._evaluation_date,
            "dev_lag": self.dev_lag(dev_lag_unit),
            **self._metadata.as_flat_dict(),
            **self._values,
        }

    def add_statics(self, source_cell, fields) -> Cell:
        """Add new static fields (e.g., `earned_premium`) to a cell."""
        static_fields = {k: v for k, v in source_cell.values.items() if k in fields}
        return self.replace(values={**self._values, **static_fields})


def values_eq(val1, val2):
    # Utility function to see if two objects of type dict[str, np.ndarray] are identical.
    if not sorted(val1.keys()) == sorted(val2.keys()):
        return False
    for k in val1.keys():
        if not np.array_equal(val1[k], val2[k]):
            return False
    return True


def format_value(val: Union[float, int], sig_figs: int = 4) -> str:
    if sig_figs < 3:
        raise Exception("sig_figs must be at least 3")
    # Prep some handy properties of val
    sign = "-" if val < 0 else ""
    aval = abs(val)
    scale = -np.inf if val == 0.0 else np.floor(np.log10(aval))
    exponent = 3 * (scale // 3)
    # If it's an integer under sig_figs, just return the integer
    if (isinstance(val, int) or val == 0.0) and scale <= sig_figs:
        return str(val)
    # Special handling of values between -1.0 and 1.0
    if scale < 0:
        zeros = int(abs(-scale)) - 1
        digits = int(round(aval * 10 ** (zeros + sig_figs)))
        return sign + "0." + ("0" * zeros) + str(digits)
    # Figure out the suffix
    if exponent == 0:
        suffix = ""
    elif exponent == 3:
        suffix = "K"
    elif exponent == 6:
        suffix = "M"
    elif exponent == 9:
        suffix = "B"
    elif exponent == 12:
        suffix = "T"
    else:
        # Switch to engineering notation for extreme values
        suffix = "e" + str(int(exponent))
    mantissa = aval / (10**exponent)
    round_places = int(sig_figs - (scale - exponent + 1))
    return sign + str(round(mantissa, round_places)) + suffix
