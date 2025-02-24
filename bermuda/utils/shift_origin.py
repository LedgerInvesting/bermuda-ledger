from ..date_utils import add_months, period_resolution
from ..triangle import Triangle

__all__ = ["shift_origin"]


def shift_origin(triangle: Triangle, origin_match_triangle: Triangle):
    """Shift the origin of a triangle to match the origin of another triangle.

    Args:
        triangle: The triangle to shift.
        origin_match_triangle: The triangle to match the origin of.

    Returns:
        Triangle: The shifted triangle.
    """
    tri_resolution = period_resolution(triangle)
    match_resolution = period_resolution(origin_match_triangle)
    if tri_resolution != match_resolution:
        raise ValueError(
            f"Cannot shift triangle with resolution {tri_resolution} to match resolution {match_resolution}"
        )
    if tri_resolution not in [3, 12]:
        raise ValueError("Only annual and quarterly triangles are supported")
    tri_origin = triangle.periods[0][0].month % tri_resolution
    match_origin = origin_match_triangle.periods[0][0].month % match_resolution
    month_shift = match_origin - tri_origin
    shifted_tri = triangle.replace(
        period_start=lambda cell: add_months(cell.period_start, month_shift),
        period_end=lambda cell: add_months(cell.period_end, month_shift),
        evaluation_date=lambda cell: add_months(cell.evaluation_date, month_shift),
    )
    return shifted_tri
