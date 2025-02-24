import datetime
from typing import Callable

import toolz as tlz

from ..base import Cell, CumulativeCell, IncrementalCell, Metadata
from ..date_utils import (
    add_months,
    dev_lag_months,
    eval_date_resolution,
    period_resolution,
    standardize_resolution,
)
from ..errors import TriangleError
from ..triangle import Triangle
from .basis import to_cumulative, to_incremental

__all__ = [
    "make_right_triangle",
    "make_right_diagonal",
    "make_pred_triangle",
    "make_pred_triangle_complement",
    "make_pred_triangle_with_init",
]


def make_right_triangle(
    triangle: Triangle[Cell], dev_lags=None, dev_lag_unit="month"
) -> Triangle:
    """Create the lower-right complement of a triangle."""

    cum_tri = to_cumulative(triangle) if triangle.is_incremental else triangle
    new_cells = tlz.mapcat(
        tlz.partial(_make_right_triangle_slice, dev_lags, dev_lag_unit),
        cum_tri.slices.values(),
    )
    right_tri = Triangle(list(new_cells))
    if triangle.is_incremental:
        right_tri = _fix_prev_evaluation_date(triangle, right_tri.to_incremental())
    return right_tri


def _make_right_triangle_slice(dev_lags, unit, slice_):
    if dev_lags is None:
        dev_lags = {cell.dev_lag(unit) for cell in slice_}

    return [
        CumulativeCell(
            period_start=cell.period_start,
            period_end=cell.period_end,
            evaluation_date=_add_dev_lag(cell, dev_lag, unit),
            metadata=cell.metadata,
            values={},
        )
        for dev_lag in dev_lags
        for cell in slice_.right_edge
        if dev_lag > cell.dev_lag(unit)
    ]


def _add_dev_lag(cell, dev_lag, unit):
    _unit = unit.lower()
    if "month" in _unit:
        return add_months(cell.period_end, dev_lag)
    elif "day" in _unit:
        return cell.period_end + datetime.timedelta(dev_lag)
    elif _unit == "timedelta":
        return cell.period_end + datetime.timedelta(dev_lag)
    else:
        raise ValueError(
            f"Unrecognized unit '{unit}', must be 'month', 'day' or 'timedelta'."
        )


def make_right_diagonal(triangle, evaluation_dates, include_historic=False) -> Triangle:
    """Create a new `Triangle` with one or more diagonals to the right of existing data."""

    cum_tri = triangle.to_cumulative() if triangle.is_incremental else triangle
    new_cells = tlz.mapcat(
        tlz.partial(_make_right_diagonal_slice, evaluation_dates, include_historic),
        cum_tri.slices.values(),
    )
    right_tri = Triangle(list(new_cells))
    if triangle.is_incremental:
        right_tri = _fix_prev_evaluation_date(triangle, right_tri.to_incremental())
    return right_tri


def _make_right_diagonal_slice(
    evaluation_dates, include_historic, slice_
) -> list[Cell]:
    max_eval = max(cell.evaluation_date for cell in slice_)
    if not include_historic:
        evaluation_dates = [
            eval_date for eval_date in evaluation_dates if eval_date > max_eval
        ]

    return [
        CumulativeCell(
            period_start=cell.period_start,
            period_end=cell.period_end,
            evaluation_date=eval_date,
            metadata=cell.metadata,
            values={},
        )
        for cell in slice_.right_edge.cells
        for eval_date in evaluation_dates
        if eval_date >= cell.period_start
    ]


def _fix_prev_evaluation_date(
    triangle: Triangle[IncrementalCell],
    right_tri: Triangle[IncrementalCell],
) -> Triangle[IncrementalCell]:
    edge_cells = []
    for slc in right_tri.slices.values():
        for period, row in slc.period_rows:
            edge_cells.append(row[0])

    fixed_edge_cells = []
    for cell in triangle.right_edge:
        fixed_edge_cells += [
            ob.replace(prev_evaluation_date=cell.evaluation_date)
            for ob in edge_cells
            if ob.period == cell.period and ob.metadata == cell.metadata
        ]

    return Triangle(
        [cell for cell in right_tri if cell not in edge_cells] + fixed_edge_cells
    )


def make_pred_triangle(
    metadata_sets: list[Metadata],
    min_period: datetime.date,
    max_period: datetime.date,
    exp_resolution: tuple[int, str],
    eval_resolution: tuple[int, str],
    exp_origin: datetime.date | None = None,
    eval_origin: datetime.date | None = None,
    min_dev_lag: tuple[int, str] | None = (0, "months"),
    max_dev_lag: tuple[int, str] | None = None,
    min_eval: datetime.date | None = None,
    max_eval: datetime.date | None = None,
    statics_fn: Callable[[Cell], dict[str, float]] | None = None,
    is_incremental: bool | None = False,
) -> Triangle:
    if exp_origin is None:
        exp_origin = min_period + datetime.timedelta(days=-1)
    if eval_origin is None:
        eval_origin = exp_origin

    # Assemble all of the experience periods in the data set
    exp_resolution_months = standardize_resolution(exp_resolution)[0]
    current_end = exp_origin
    periods = []
    while current_end < max_period:
        next_begin = current_end + datetime.timedelta(days=1)
        next_end = add_months(current_end, exp_resolution_months)
        if next_end <= max_period:
            periods.append((next_begin, next_end))
        current_end = next_end

    # Assemble all of the evaluation dates in the data set
    eval_resolution_months = standardize_resolution(eval_resolution)[0]
    if max_eval is None and max_dev_lag is None:
        raise Exception("One of `max_eval` or `max_dev_lag` must be specified")
    if max_eval is None:
        max_dev_lag_months = standardize_resolution(max_dev_lag)[0]
        max_eval = add_months(periods[-1][1], max_dev_lag_months)
    if max_dev_lag is None:
        max_dev_lag_months = dev_lag_months(periods[0][1], max_eval)
    else:
        max_dev_lag_months = standardize_resolution(max_dev_lag)[0]
    min_dev_lag_months = standardize_resolution(min_dev_lag)[0]
    if min_eval is None:
        min_eval = add_months(periods[0][1], min_dev_lag_months)

    current_eval = min_eval
    eval_dates = []
    while current_eval <= max_eval:
        eval_dates.append(current_eval)
        current_eval = add_months(current_eval, eval_resolution_months)

    def _cell_with_statics(period_start, period_end, evaluation_date, metadata):
        ob = CumulativeCell(
            period_start=period_start,
            period_end=period_end,
            evaluation_date=evaluation_date,
            metadata=metadata,
            values={},
        )

        try:
            return ob.replace(values=statics_fn(ob))
        except (KeyError, IndexError):
            return None

    pred_cells = [
        _cell_with_statics(period_start, period_end, evaluation_date, metadata)
        for period_start, period_end in periods
        for evaluation_date in eval_dates
        for metadata in metadata_sets
        if min_dev_lag_months
        <= dev_lag_months(period_end, evaluation_date)
        <= max_dev_lag_months
    ]
    pred_triangle = Triangle([cell for cell in pred_cells if cell is not None])

    if is_incremental:
        pred_triangle = pred_triangle.to_incremental()
    return pred_triangle


def make_pred_triangle_complement(
    init_triangle: Triangle[Cell],
    static_fields: list[str] | None = None,
    max_dev_lag: int | None = None,
    eval_date_resolution_override: int | None = None,
) -> Triangle:
    evaluation_date_res = (
        eval_date_resolution(init_triangle)
        if eval_date_resolution_override is None
        else eval_date_resolution_override
    )
    if evaluation_date_res is None:
        raise TriangleError(
            "Must provide `init_triangle` with different evaluation dates, or the evaluation date "
            "resolution of the triangle manually via. eval_date_resolution_override."
        )

    static_fields = (
        static_fields if static_fields else ["earned_premium", "earned_exposure"]
    )

    def statics_fn(empty_cell: Cell) -> dict[str, float]:
        edge_cell = [
            cell
            for cell in init_triangle.right_edge.cells
            if cell.details == empty_cell.details
            and cell.period_start == empty_cell.period_start
        ][0]
        return {k: v for k, v in edge_cell.values.items() if k in static_fields}

    if max_dev_lag is None:
        max_dev_lag = init_triangle.dev_lags()[-1]

    raw_cells = make_pred_triangle(
        metadata_sets=init_triangle.metadata,
        min_period=init_triangle.periods[0][0],
        max_period=init_triangle.periods[-1][-1],
        exp_resolution=(period_resolution(init_triangle), "months"),
        eval_resolution=(evaluation_date_res, "months"),
        exp_origin=init_triangle.periods[0][0] - datetime.timedelta(days=1),
        eval_origin=init_triangle.evaluation_dates[0] - datetime.timedelta(days=1),
        min_dev_lag=(init_triangle.dev_lags()[0], "months"),
        max_dev_lag=(max_dev_lag, "months"),
        min_eval=init_triangle.evaluation_dates[0],
        statics_fn=statics_fn,
        is_incremental=init_triangle.is_incremental,
    )

    init_cells = {}
    for cell in init_triangle:
        frozen_details = frozenset(cell.details.items())
        if frozen_details not in init_cells:
            init_cells[frozen_details] = []
        init_cells[frozen_details].append((cell.period_start, cell.evaluation_date))
    right_edge_start = {
        slice_period_id: cell[-1].evaluation_date
        for slice_period_id, cell in init_triangle.slice_period_rows
    }

    return Triangle(
        [
            cell
            for cell in raw_cells
            if (
                (cell.period_start, cell.evaluation_date)
                not in init_cells[frozenset(cell.details.items())]
            )
            and cell.evaluation_date
            > right_edge_start.get((cell.metadata, cell.period), datetime.date.min)
        ]
    )


def make_pred_triangle_with_init(
    init_triangle: Triangle[Cell],
    pred_triangle: Triangle[Cell] | None = None,
    max_dev_lag: tuple[int, str] | None = None,
    eval_resolution: tuple[int, str] | None = None,
    max_eval_date: datetime.date | None = None,
) -> Triangle:
    # Return pred_triangle if provided and no conflicts
    if pred_triangle is not None:
        if max_eval_date is not None:
            raise TriangleError(
                "`pred_triangle` and `max_eval_date` cannot both be specified"
            )
        if max_dev_lag is not None:
            raise TriangleError(
                "`pred_triangle` and `max_dev_lag` cannot both be specified"
            )
        if eval_resolution is not None:
            raise TriangleError(
                "`pred_triangle` and `eval_resolution` cannot both be specified"
            )
        if type(init_triangle.cells[0]) != type(pred_triangle.cells[0]):  # noqa: E721
            raise TriangleError(
                "`init_triangle` and `pred_triangle` have different cell types"
            )
        return pred_triangle

    # Fill in default values for other args if not supplied
    if max_dev_lag is None:
        raise TriangleError(
            "`max_dev_lag` must be supplied if `pred_triangle` is missing"
        )
    if eval_resolution is None:
        raise TriangleError(
            "`eval_resolution` must be supplied if `pred_triangle` is missing"
        )
    if max_eval_date is None:
        max_eval_date = datetime.date.max

    dev_lags = _get_all_lag_months(eval_resolution, max_dev_lag)
    pred_tri = init_triangle.make_right_triangle(dev_lags)
    return pred_tri.filter(lambda cell: cell.evaluation_date <= max_eval_date)


def _get_all_lag_months(eval_resolution, max_dev_lag):
    standardized_resolution = standardize_resolution(eval_resolution)
    standardized_max_lag = standardize_resolution(max_dev_lag)
    if standardized_resolution[1] != "month":
        raise TriangleError(
            "Cannot construct a `pred_triangle` with non-monthly resolution"
        )
    if standardized_max_lag[1] != "month":
        raise TriangleError(
            "Cannot construct a `pred_triangle` with non-monthly max dev lag"
        )
    lag_months = []
    lag = 0
    while lag <= standardized_max_lag[0]:
        lag_months.append(lag)
        lag += standardized_resolution[0]
    return lag_months
