import datetime
from collections import defaultdict

from .. import date_utils
from ..base import Cell, CumulativeCell
from ..errors import TriangleError
from ..triangle import Triangle
from .basis import to_cumulative, to_incremental
from .summarize import summarize_cell_values

__all__ = [
    "aggregate",
]


def aggregate(
    triangle: Triangle[Cell],
    period_resolution: tuple[int, str] | None = None,
    eval_resolution: tuple[int, str] | None = None,
    period_origin: datetime.date = datetime.date(1999, 12, 31),
    eval_origin: datetime.date = datetime.date(1999, 12, 31),
    summarize_premium: bool = True,
) -> Triangle[Cell]:
    """Change the period and/or evaluation resolution of a triangle.

    Period resolution is defined as the length of each experience period in a triangle.
    Similarly, evaluation resolution is defined as the interval between each successive
    evaluation date in a triangle. Period resolution and evaluation are commonly equal to
    each other, but this is not a hard requirement.

    This aggregation implementation will not try to interpolate, apportion, or otherwise
    estimate triangle cells when data is insufficient to determine an exact answer. This
    means that if a triangle has two experience periods, `2020-01-01:2020-09-30` and
    `2020-10-01:2021-06-30`, aggregation to annual data is not possible.

    Aggregation over experience periods is equivalent to summing all values from all cells
    that are being merged into a new cell. No warnings are issued if not all cells have the
    same set of attributes, or if the new experience period isn't completely covered by
    merged cells. Depending on the semantics of your source data and intended application,
    this may not be advisable.

    Aggregation over evaluation dates is equivalent to filtering out evaluation dates.
    No effort is made to "carry forward" cells with gaps in their evaluation date history.
    The aggregation strategy is to sum fields over cells. This makes sense for fields such as
    paid loss, earned premium, and claim counts, but aggregation-by-summation doesn't make
    sense for fields such as loss ratio or average claim severity. This function does not
    attempt to infer an appropriate aggregation strategy based on the name of each field.

    Args:
        triangle: Triangle to aggregate.
        period_resolution: If supplied, the length of each experience perid. Format is a tuple
            with an integer number of units and a string specifying the unit. E.g.,
            `(7, "day")`, `(3, "months")`, `(1, "year")`. If missing, no experience period
            aggregation will be performed.
        eval_resolution: If supplied, the interval between each evaluation date. Format is
            the same as for `period_resolution`. If missing, no evaluation date aggregation
            will be performed.
        period_origin: A date that will be guaranteed to be a period start date (if there
            is any contemporaneous data). This argument is used for offseting experience
            periods temporally: e.g., for annual experience periods that start on July 1
            of each year, supply `period_origin=datetime.date(2000, 7, 1)` or similar.
        eval_origin: A date that will be guaranteed to be an evaluation date (if there
            is any contemporaneous data). See `period_origin` for details of semantics.
        summarize_premium: Whether or not to sum up premiums over aggregated cells. Defaults
            to True, set to False when summarizing over loss details over which premiums should
            not be summed.
    """
    cum_triangle = to_cumulative(triangle) if triangle.is_incremental else triangle
    agg_slices = []
    for slice_ in cum_triangle.slices.values():
        eval_agg_slice = _aggregate_eval(slice_, eval_resolution, eval_origin)
        agg_slice = _aggregate_period(
            eval_agg_slice, period_resolution, period_origin, summarize_premium
        )
        agg_slices.append(agg_slice)

    aggregate_triangle = (
        to_incremental(sum(agg_slices)) if triangle.is_incremental else sum(agg_slices)
    )

    return aggregate_triangle


def _aggregate_eval(triangle, eval_resolution, eval_origin):
    if eval_resolution is None:
        return triangle

    resolution = date_utils.standardize_resolution(eval_resolution)
    first_eval = triangle.evaluation_dates[0]
    last_eval = triangle.evaluation_dates[-1]

    # Move current_eval to on or before the first evaluation date in the slice
    current_eval = eval_origin
    while date_utils.resolution_delta(current_eval, resolution) < first_eval:
        current_eval = date_utils.resolution_delta(current_eval, resolution)
    while current_eval >= first_eval:
        current_eval = date_utils.resolution_delta(
            current_eval, resolution, negative=True
        )

    # Construct the set of all valid evaluations
    valid_evals = []
    current_eval = date_utils.resolution_delta(current_eval, resolution)
    while current_eval <= last_eval:
        valid_evals.append(current_eval)
        current_eval = date_utils.resolution_delta(current_eval, resolution)

    # Filter for only the valid evaluation dates
    return Triangle(
        [cell for cell in triangle.cells if cell.evaluation_date in valid_evals]
    )


# noinspection PyShadowingNames
def _aggregate_period(triangle, period_resolution, period_origin, summarize_premium):
    if period_resolution is None:
        return triangle

    resolution = date_utils.standardize_resolution(period_resolution)
    cells = sorted(triangle.cells, key=lambda cell: cell.coordinates)
    first_start = cells[0].period_start

    # Find the period_end immediately before the start of the aggregated triangle
    current_init = period_origin
    while date_utils.resolution_delta(current_init, resolution) < first_start:
        current_init = date_utils.resolution_delta(current_init, resolution)
    while current_init >= first_start:
        current_init = date_utils.resolution_delta(
            current_init, resolution, negative=True
        )

    # Map from old periods to new periods
    new_coordinate_cells = defaultdict(list)
    current_start = current_init + datetime.timedelta(days=1)
    current_end = date_utils.resolution_delta(current_init, resolution)
    for cell in cells:
        # We can step forward safely like this because we're sorted on period_start
        while current_end < cell.period_start:
            current_init = date_utils.resolution_delta(current_init, resolution)
            current_start = current_init + datetime.timedelta(days=1)
            current_end = date_utils.resolution_delta(current_init, resolution)

        # We know cell.period_start is in between current_start and current_end, we just
        # need to check for border crossings on the far side.
        if cell.period_end > current_end:
            raise TriangleError(
                f"Cannot aggregate periods to f{period_resolution} with "
                f"origin {period_origin}; triangle periods cross aggregate period bounds"
            )

        # Create a new cell to put in the coordinate pile
        new_cell = Cell(
            period_start=current_start,
            period_end=current_end,
            evaluation_date=cell.evaluation_date,
            values=cell.values,
            metadata=cell.metadata,
        )
        new_coordinate_cells[new_cell.coordinates].append(new_cell)

    # Aggregate the new cells
    new_cells = [
        CumulativeCell(
            period_start=cells[0].period_start,
            period_end=cells[0].period_end,
            evaluation_date=cells[0].evaluation_date,
            values=summarize_cell_values(cells, summarize_premium=summarize_premium),
            metadata=cells[0].metadata,
        )
        for cells in new_coordinate_cells.values()
    ]
    return Triangle(new_cells)
