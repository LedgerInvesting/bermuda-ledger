import copy
import datetime
from collections import defaultdict

import toolz as tlz

from ..base import CumulativeCell, IncrementalCell
from ..date_utils import add_months, id_to_month, month_to_id, resolution_delta
from ..errors import TriangleError
from ..triangle import Triangle

__all__ = [
    "to_incremental",
    "to_cumulative",
]


def to_incremental(
    triangle: Triangle[CumulativeCell],
) -> Triangle[IncrementalCell]:
    """Convert a Triangle from cumulative to incremental basis."""

    if triangle.is_incremental:
        return triangle

    grouped_cells = tlz.groupby(lambda ob: (ob.period, ob.metadata), triangle.cells)
    ordered_cells = tlz.valmap(sorted, grouped_cells)

    incremental_cells = []
    for ((period_start, period_end), metadata), cells in ordered_cells.items():
        # Add the first cell
        incremental_cells.append(
            IncrementalCell(
                period_start=period_start,
                period_end=period_end,
                prev_evaluation_date=resolution_delta(period_start, (-1, "days")),
                evaluation_date=cells[0].evaluation_date,
                metadata=metadata,
                values=copy.deepcopy(cells[0].values),
            )
        )

        # Add subsequent cells
        for prev_cell, next_cell in zip(cells[:-1], cells[1:]):
            incremental_cells.append(
                IncrementalCell(
                    period_start=period_start,
                    period_end=period_end,
                    prev_evaluation_date=prev_cell.evaluation_date,
                    evaluation_date=next_cell.evaluation_date,
                    metadata=metadata,
                    values=_values_diff(prev_cell.values, next_cell.values),
                )
            )

    return Triangle(incremental_cells)


def to_cumulative(
    triangle: Triangle[IncrementalCell],
) -> Triangle[CumulativeCell]:
    """Convert a Triangle from incremental to cumulative basis."""

    if not triangle.is_incremental:
        return triangle

    grouped_cells = tlz.groupby(lambda ob: (ob.period, ob.metadata), triangle.cells)
    ordered_cells = tlz.valmap(sorted, grouped_cells)

    cumulative_cells = []
    for ((period_start, period_end), metadata), cells in ordered_cells.items():
        if (
            resolution_delta(cells[0].prev_evaluation_date, (1, "days"))
            != cells[0].period_start
        ):
            raise TriangleError(
                "Cannot convert incomplete incremental triangle to cumulative"
            )

        current_values = copy.deepcopy(cells[0].values)
        current_evaluation_date = cells[0].evaluation_date
        cumulative_cells.append(
            CumulativeCell(
                period_start=period_start,
                period_end=period_end,
                evaluation_date=cells[0].evaluation_date,
                metadata=metadata,
                values=copy.deepcopy(current_values),
            )
        )

        for cell in cells[1:]:
            if cell.prev_evaluation_date != current_evaluation_date:
                raise TriangleError(
                    "Cannot convert incomplete incremental triangle to cumulative"
                )
            current_evaluation_date = cell.evaluation_date
            current_values = _values_add(current_values, cell.values)
            cumulative_cells.append(
                CumulativeCell(
                    period_start=period_start,
                    period_end=period_end,
                    evaluation_date=cell.evaluation_date,
                    metadata=metadata,
                    values=current_values,
                )
            )

    return Triangle(cumulative_cells)


def accident_quarter_to_policy_year(
    accident_quarter_tri: Triangle,
    policy_length_months: int = 12,
    policy_year_origin: datetime.date = datetime.date(2020, 1, 1),
    continuous_issuance: bool = True,
) -> Triangle:
    """Converts accident quarter triangles to policy year triangles.

    Note this function will fail if the evaluation dates are not consistent for the whole policy
    period.

    Args:
        accident_quarter_tri: The triangle to convert.
        policy_length_months: The length of the policy in months.
        policy_year_origin: The date of the start of the policy year.
        continuous_issuance: Whether the policy is issued continuously or not. If not, all premium
        is written in the first month of the policy year.
    """
    results = Triangle([])
    for _, tri_slice in accident_quarter_tri.slices.items():
        results += _accident_quarter_to_policy_year_slice(
            tri_slice,
            policy_length_months=policy_length_months,
            policy_year_origin=policy_year_origin,
            continuous_issuance=continuous_issuance,
        )
    return results


def _accident_quarter_to_policy_year_slice(
    accident_quarter_tri_slice: Triangle,
    policy_length_months: int = 12,
    policy_year_origin: datetime.date = datetime.date(2020, 1, 1),
    continuous_issuance: bool = True,
) -> Triangle:
    """Converts accident quarter triangles to policy year triangles.

    Note this function will fail if the evaluation dates are not consistent for the whole policy
    period.

    Args:
        accident_quarter_tri: The triangle to convert.
        policy_length_months: The length of the policy in months.
        policy_year_origin: The date of the start of the policy year.
        continuous_issuance: Whether the policy is issued continuously or not. If not, all premium
        is written in the first month of the policy year.
    """

    if len(accident_quarter_tri_slice.metadata) > 1:
        raise ValueError("The accident quarter triangle should have only one slice.")
    if len(accident_quarter_tri_slice.right_edge.evaluation_dates) > 1:
        raise ValueError(
            "The accident quarter triangle has more than one evaluation date at the right edge. \n"
            "This will could lead to undercounting policy-basis premium and loss."
        )

    policy_years = policy_years_covered(accident_quarter_tri_slice, policy_year_origin)
    policy_year_ep_shares = {}
    for policy_year in policy_years:
        policy_year_ep_shares[policy_year] = monthly_ep_to_quarterly_ep(
            _policy_earned_premium_share_by_month(
                risk_start_date=policy_year[0],
                risk_end_date=policy_year[1],
                policy_length_months=policy_length_months,
                continuous_issuance=continuous_issuance,
            ),
            accident_quarter_tri_slice,
        )
    accident_quarter_py_shares = defaultdict(dict)
    for py, aq_share in policy_year_ep_shares.items():
        for aq, share in aq_share.items():
            accident_quarter_py_shares[aq].update({py: share})

    # normalize the shares for each accident quarter
    for _, py_shares in accident_quarter_py_shares.items():
        total_weight = sum(py_shares.values())
        for py, share in py_shares.items():
            py_shares[py] = share / total_weight

    cells = []
    for policy_period in policy_years:
        for evaluation_date in accident_quarter_tri_slice.evaluation_dates:
            aq_cells = accident_quarter_tri_slice[:, evaluation_date, :]
            vals_dict = defaultdict(float)
            for cell in aq_cells:
                py_share = accident_quarter_py_shares[cell.period]
                if policy_period in py_share:
                    for field, val in cell.values.items():
                        vals_dict[field] += val * py_share[policy_period]
            if vals_dict:
                cells.append(
                    CumulativeCell(
                        period_start=policy_period[0],
                        period_end=policy_period[1],
                        evaluation_date=evaluation_date,
                        values=vals_dict,
                        metadata=cell.metadata,
                    )
                )

    return Triangle(cells).derive_metadata(risk_basis="Policy")


def _policy_earned_premium_share_by_month(
    risk_start_date: datetime.date,
    risk_end_date: datetime.date,
    policy_length_months: int,
    continuous_issuance: bool,
    monthly_earning_pattern: dict[datetime.date, float] | None = None,
) -> dict[datetime.date, float]:
    written_start_month_id = month_to_id(risk_start_date)
    if continuous_issuance:
        written_end_month_id = month_to_id(risk_end_date)
    else:
        written_end_month_id = written_start_month_id

    earned_premium_by_month = {
        month_id: 0
        for month_id in range(
            written_start_month_id, written_end_month_id + policy_length_months + 1
        )
    }
    if monthly_earning_pattern:
        for month_id in earned_premium_by_month:
            earned_premium_by_month[month_id] = {
                month_to_id(date): value
                for date, value in monthly_earning_pattern.items()
            }.get(month_id, 0)
    else:
        written_months = written_end_month_id - written_start_month_id + 1
        monthly_volume = 1 / written_months / policy_length_months
        for written_month_id in range(written_start_month_id, written_end_month_id + 1):
            earned_premium_by_month[written_month_id] += monthly_volume / 2
            for earn_offset in range(1, policy_length_months):
                earned_premium_by_month[written_month_id + earn_offset] += (
                    monthly_volume
                )
            earned_premium_by_month[written_month_id + policy_length_months] += (
                monthly_volume / 2
            )

    return {id_to_month(id_): value for id_, value in earned_premium_by_month.items()}


def policy_years_covered(
    accident_basis_tri: Triangle, policy_year_origin: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    """Returns the policy years covered by the accident-basis triangle."""
    first_accident_period_start = accident_basis_tri.periods[0][0]
    last_accident_period_end = accident_basis_tri.periods[-1][1]
    py_start = datetime.date(
        first_accident_period_start.year,
        policy_year_origin.month,
        policy_year_origin.day,
    )
    if py_start > first_accident_period_start:
        py_start = datetime.date(
            first_accident_period_start.year - 1,
            policy_year_origin.month,
            policy_year_origin.day,
        )
    period_starts = [py_start]
    while py_start < last_accident_period_end:
        py_start = add_months(py_start, 12)
        period_starts.append(py_start)
    return [
        (start, add_months(start, 12) + datetime.timedelta(days=-1))
        for start in period_starts
    ]


def monthly_ep_to_quarterly_ep(
    policy_month_ep_share: dict[datetime.date, float], accident_tri: Triangle
) -> dict[tuple, float]:
    """Aggregates the monthly EP share dictionary to match the quarterly periods."""
    accident_quarter_ep_share = defaultdict(float)
    for month, share in policy_month_ep_share.items():
        for quarter in accident_tri.periods:
            if month >= quarter[0] and month <= quarter[1]:
                accident_quarter_ep_share[quarter] += share
    return accident_quarter_ep_share


def _values_diff(prev_values, next_values):
    prev_keys = set(prev_values.keys())
    next_keys = set(next_values.keys())
    if prev_keys.symmetric_difference(next_keys):
        raise TriangleError("Consecutive cells have different value keys")

    return {
        k: (
            next_values[k] if k == "earned_premium" else next_values[k] - prev_values[k]
        )
        for k in prev_keys
    }


def _values_add(curr_values, next_values):
    curr_keys = set(curr_values.keys())
    next_keys = set(next_values.keys())
    if curr_keys.symmetric_difference(next_keys):
        raise TriangleError("Consecutive cells have different value keys")

    return {
        k: (
            next_values[k] if k == "earned_premium" else curr_values[k] + next_values[k]
        )
        for k in curr_keys
    }
