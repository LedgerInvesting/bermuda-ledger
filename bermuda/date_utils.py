from __future__ import annotations

import calendar
import datetime
import math
from collections import Counter, defaultdict

import numpy as np

__all__ = [
    "DevLag",
    "calculate_dev_lag",
    "add_months",
    "dev_lag_months",
    "standardize_resolution",
    "resolution_delta",
    "month_to_id",
    "id_to_month",
    "period_resolution",
    "eval_date_resolution",
    "is_triangle_monthly",
    "drop_off_diagonals",
]


DevLag = int | float | datetime.timedelta


def calculate_dev_lag(
    period_end: datetime.date, evaluation_date: datetime.date, unit: str = "months"
) -> DevLag:
    """Compute the development lag of the cell in the specified units."""
    _unit = unit.lower()

    # Short-circuit handling of maximum evaluation dates
    if evaluation_date == datetime.date.max:
        if _unit == "timedelta":
            return datetime.timedelta.max
        else:
            return np.inf

    # Base case
    if "month" in _unit:
        return dev_lag_months(period_end, evaluation_date)
    elif "day" in _unit:
        return (evaluation_date - period_end).days
    elif _unit == "timedelta":
        return evaluation_date - period_end
    else:
        raise ValueError(
            f"Unrecognized unit '{unit}', must be 'month', 'day' or 'timedelta'."
        )


def add_months(dt: datetime.date, delta: float) -> datetime.date:
    """Add a floating-point number of months to a date and return a date."""
    # Short-circuit handling of infinite deltas:
    if delta == np.inf:
        return datetime.date.max

    # Compute result as an offset from a fixed origin date
    init_lag = dev_lag_months(datetime.date(1969, 12, 31), dt)
    final_lag = init_lag + delta

    # Split the result into an integer and fractional part, with the convention that the
    # fractional part is in the domain (0.0, 1.0].
    final_lag_months = int(final_lag) - 1 if final_lag % 1 == 0.0 else int(final_lag)
    final_lag_frac = 1.0 if final_lag % 1 == 0.0 else final_lag % 1

    # Figure out the calendar year, month, and day of the result.
    month = final_lag_months % 12 + 1
    year = 1970 + final_lag_months // 12
    _, days_in_month = calendar.monthrange(year, month)
    day = round(final_lag_frac * days_in_month)

    # Special-casing required if day-of-month is zero.
    if day == 0:
        return datetime.date(year, month, 1) - datetime.timedelta(days=1)
    else:
        return datetime.date(year, month, day)


def _month_fraction(dt: datetime.date) -> float:
    """Compute the fractional portion of the month that has elapsed."""
    _, last_day_of_month = calendar.monthrange(dt.year, dt.month)
    return dt.day / last_day_of_month


def dev_lag_months(start: datetime.date, stop: datetime.date) -> float:
    """Compute the floating-point number of months between two dates."""
    start_frac = _month_fraction(start)
    stop_frac = _month_fraction(stop)
    delta_years = stop.year - start.year
    delta_months = stop.month - start.month
    # adding 0.019 to a 31 day month is the smallest value to increment by one day
    return 12 * delta_years + delta_months - start_frac + stop_frac


def standardize_resolution(resolution):
    # Convert resolution to a standard format -- units are either "day" or "month"
    quantity, units = resolution
    clean_units = units.lower()
    if "month" in clean_units:
        clean_units = "month"
    elif "quarter" in clean_units:
        quantity *= 3
        clean_units = "month"
    elif "year" in clean_units:
        quantity *= 12
        clean_units = "month"
    elif "day" in clean_units:
        clean_units = "day"
    elif "week" in clean_units:
        quantity *= 7
        clean_units = "day"
    else:
        raise ValueError(f"Unrecognized resolution unit {units}")
    return quantity, clean_units


def resolution_delta(date, resolution, negative=False):
    # Add (or subtract) a specified resolution from a given date.
    quantity, units = resolution
    if negative:
        quantity *= -1
    if units == "month":
        return add_months(date, quantity)
    else:
        return date + datetime.timedelta(days=quantity)


def month_to_id(dt: datetime.date) -> int:
    """Convert a date to an integer representing the month containing the date."""
    return 12 * (dt.year - 1970) + dt.month - 1


def id_to_month(id: int, beginning=True) -> datetime.date:
    """Convert an integer representing a month to the first date in the month."""
    if beginning:
        return datetime.date(1970 + id // 12, id % 12 + 1, 1)
    else:
        id += 1
        return datetime.date(1970 + id // 12, id % 12 + 1, 1) - datetime.timedelta(
            days=1
        )


def period_resolution(tri: Triangle) -> int:  # noqa: F821
    """Determine the biggest unit of resolution (expressed in months) that can losslessly
    represent all experience periods in a triangle."""
    period_starts, period_ends = zip(*tri.periods)
    start_months = [month_to_id(start) for start in period_starts]
    next_start_months = [month_to_id(end) + 1 for end in period_ends]
    starts_ordered = sorted(set(start_months + next_start_months))
    start_diffs = _diff(starts_ordered)
    if not start_diffs:
        return None
    return _multi_gcd(start_diffs)


def eval_date_resolution(tri: Triangle) -> int:  # noqa: F821
    """Determine the biggest unit of resolution (expressed in months) that can losslessly
    represent all evaluation dates in a triangle."""
    eval_months = [month_to_id(eval_dt) for eval_dt in tri.evaluation_dates]
    evals_ordered = sorted(eval_months)
    eval_diffs = _diff(evals_ordered)
    if not eval_diffs:
        return None
    return _multi_gcd(eval_diffs)


def drop_off_diagonals(triangle: Triangle) -> Triangle:  # noqa: F821
    """Drop off-diagonal values from a triangle.

    Only keep the most common resolution and diagonals with the most
    popular origin at that resolution.
    """
    from_previous = defaultdict(list)
    to_next = defaultdict(list)
    diffs = []
    for prev_date, next_date in zip(
        triangle.evaluation_dates[:-1], triangle.evaluation_dates[1:]
    ):
        diff = calculate_dev_lag(prev_date, next_date)
        diffs.append(diff)
        from_previous[diff].append(next_date)
        to_next[diff].append(prev_date)
    most_common_resolution = Counter(diffs).most_common(1)[0][0]
    eval_origins = [
        month_to_id(date) % most_common_resolution for date in triangle.evaluation_dates
    ]
    most_common_origin = Counter(eval_origins).most_common(1)[0][0]
    most_useful_evals = [
        date
        for date, origin in zip(triangle.evaluation_dates, eval_origins)
        if origin == most_common_origin
    ]
    return triangle.filter(lambda cell: cell.evaluation_date in most_useful_evals)


def is_triangle_monthly(tri: Triangle) -> bool:  # noqa: F821
    """Test if all periods and evaluation dates in a triangle are in monthly increments."""
    period_starts, period_ends = zip(*tri.periods)
    all_starts_aligned = all([_is_month_start(start) for start in period_starts])
    all_ends_aligned = all([_is_month_end(end) for end in period_ends])
    all_evals_aligned = all(
        [_is_month_end(eval_dt) for eval_dt in tri.evaluation_dates]
    )
    return all_starts_aligned and all_ends_aligned and all_evals_aligned


def _is_month_start(dt: datetime.date) -> bool:
    """Test if a date is at the beginning of a month."""
    return dt.day == 1


def _is_month_end(dt: datetime.date) -> bool:
    """Test if a date is at the end of a month."""
    next_day = dt + datetime.timedelta(days=1)
    return next_day.day == 1


def _multi_gcd(xs: list[int]) -> int:
    """Compute the gcd of a list of integers."""
    unique_xs = list(set(xs))
    if len(unique_xs) == 1:
        return unique_xs[0]
    # Reduce over unique xs
    result = math.gcd(unique_xs[0], unique_xs[1])
    for x in unique_xs[2:]:
        result = math.gcd(result, x)
    return result


def _diff(xs: list[int]) -> list[int]:
    """Compute the differences between a list of integers."""
    return [after - before for before, after in zip(xs[:-1], xs[1:])]
