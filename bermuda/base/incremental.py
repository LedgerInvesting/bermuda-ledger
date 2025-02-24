import datetime
from collections import namedtuple
from typing import Any, Tuple, Union

import numpy as np

from ..date_utils import calculate_dev_lag
from .cell import Cell, CellValue, format_value
from .metadata import Metadata

__all__ = [
    "CumulativeCell",
    "IncrementalCell",
]


_constr_coordinates = namedtuple(
    "coordinates",
    ["period_start", "period_end", "evaluation_date", "prev_evaluation_date"],
)


class CumulativeCell(Cell):
    """Subclass for cumulative cells.

    The base Cell class is already cumulative, so this subclass mostly exists for the sake of the
    type system. Cell implies cumulative or incremental; CumulativeCell implies that it doesn't
    work with incremental cells.
    """

    pass


class IncrementalCell(Cell):
    def __init__(
        self,
        period_start: datetime.date,
        period_end: datetime.date,
        prev_evaluation_date: datetime.date,
        evaluation_date: datetime.date,
        values: dict[str, CellValue],
        metadata: Metadata = None,
        _skip_validation: bool = False,
    ):
        super().__init__(
            period_start=period_start,
            period_end=period_end,
            evaluation_date=evaluation_date,
            values=values,
            metadata=metadata,
            _skip_validation=_skip_validation,
        )
        if evaluation_date <= prev_evaluation_date and not _skip_validation:
            raise ValueError(
                f"`evaluation_date` ({evaluation_date}) must be > `prev_evaluation_date` ({prev_evaluation_date})."
            )
        self._prev_evaluation_date = prev_evaluation_date

    prev_evaluation_date = property(lambda self: self._prev_evaluation_date)

    def __eq__(self, other: Cell) -> bool:
        return (
            super().__eq__(other)
            and self._prev_evaluation_date == other._prev_evaluation_date
        )

    def __lt__(self, other: Cell) -> bool:
        """Required so that lists of cells can be sortable.
        Comparison is based on (period_start, period_end, evaluation_date, prev_evaluation_date) tuple.
        """
        return (
            self._metadata,
            self._period_start,
            self._period_end,
            self._evaluation_date,
            self._prev_evaluation_date,
        ) < (
            other._metadata,
            other._period_start,
            other._period_end,
            other._evaluation_date,
            other._prev_evaluation_date,
        )

    @property
    def coordinates(
        self,
    ) -> Tuple[datetime.date, datetime.date, datetime.date, datetime.date]:
        """Location of the cell: period_start, period_end, evaluation_date, prev_evaluation_date."""
        return _constr_coordinates(
            self._period_start,
            self._period_end,
            self._evaluation_date,
            self._prev_evaluation_date,
        )

    def eval_lag(self, unit: str = "months") -> Union[float, int, datetime.timedelta]:
        """Compute the evaluation lag of the cell in the specified units.

        Args:
            unit (str): One of 'month', 'day', 'timedelta'.  'month' returns eval_lag as
                a float number of months, day returns dev_lag as an integer number
                of days, and 'timedelta' returns a datetime.timedelta object.
        """
        return calculate_dev_lag(
            self._prev_evaluation_date, self._evaluation_date, unit
        )

    def to_record(self, dev_lag_unit="month") -> dict[str, Any]:
        """Return dict representation of Cell."""
        return {
            "period_start": self._period_start,
            "period_end": self._period_end,
            "evaluation_date": self._evaluation_date,
            "prev_evaluation_date": self._prev_evaluation_date,
            "dev_lag": self.dev_lag(dev_lag_unit),
            **self._metadata.as_flat_dict(),
            **self._values,
        }

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
            f"    <tr><th> Previous Evaluation Date: </th><td> {self.prev_evaluation_date} </td></tr>",
            f"    <tr><th> Development Lag: </th><td> {self.dev_lag()} months </td></tr>",
            '    <tr><th colspan=2 style="text-align:center"> Values </th></tr>',
            *value_rows,
            '    <tr><th colspan=2 style="text-align:center"> Metadata </th></tr>',
            *metadata_rows,
            "  </tbody></table>",
        ]
        return "\n".join(table)
