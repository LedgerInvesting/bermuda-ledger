from ..date_utils import add_months
from ..triangle import Triangle


def fill_forward_gaps(
    triangle: Triangle,
    eval_resolution: int | None = None,
    fill_with_none: bool = False,
) -> Triangle:
    """For each slice period, fill forward cells for missing evaluations
    at the current triangle resolution.

    Args:
        triangle: Triangle to fill
        eval_resolution: Evaluation resolution in months, if not supplied
            this will be calculated from the existing triangle
        fill_with_none: Fill missing cell values with None, otherwise
            copy the previous cell values
    """
    if eval_resolution is None:
        eval_resolution = triangle.eval_date_resolution
    filled_cells = []
    for _, period in triangle.period_rows:
        period_cells = {cell.dev_lag(): cell for cell in period}
        required_lags = {
            lag
            for lag in range(
                int(period[0].dev_lag()),
                int(period[-1].dev_lag() + eval_resolution),
                eval_resolution,
            )
        }
        new_lags = required_lags - set(period_cells.keys())
        for lag in sorted(new_lags):
            period_cells[lag] = period_cells[lag - eval_resolution].replace(
                evaluation_date=lambda cell: add_months(cell.period_end, lag),
            )
            if fill_with_none:
                period_cells[lag] = period_cells[lag].replace(
                    values={k: None for k in period_cells[lag].values}
                )
        filled_cells.extend(period_cells.values())

    return Triangle(filled_cells)
