from ..date_utils import add_months
from ..triangle import Triangle


def backfill(
    triangle: Triangle,
    static_fields: list[str] = ["earned_premium"],
    eval_resolution: int | None = None,
    min_dev_lag: int = 0,
) -> Triangle:
    """Backfill missing cells for each period with $0 loss, 0 claim counts
    and the first value of earned premium.

    Args:
        triangle: Triangle with missing cells.
        static_fields: Fields for which values will be preseved in the backfill.
        eval_resolution: Resolution of the evaluation date. If None, it will be
            calculated from the triangle.
        min_dev_lag: Minimum development lag for the backfilled cells. Can be negative,
            but must be greater than -period_resolution.
    """
    if eval_resolution is None:
        eval_resolution = triangle.eval_date_resolution
    additional_cells = []
    for _, period in triangle.period_rows:
        first_cell = period[0]
        replacement_values = {k: 0 for k in first_cell.values.keys()}
        for field in static_fields:
            replacement_values[field] = first_cell.values[field]
        current_lag = first_cell.dev_lag()
        while current_lag > min_dev_lag:
            current_lag -= eval_resolution
            additional_cells.append(
                first_cell.replace(
                    evaluation_date=lambda cell: add_months(
                        cell.period_end, current_lag
                    ),
                    values=replacement_values.copy(),
                )
            )

    return triangle + Triangle(additional_cells)
