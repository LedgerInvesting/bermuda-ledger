import warnings

import numpy as np

from ..base import Cell
from ..date_utils import dev_lag_months, month_to_id
from ..triangle import Triangle
from .basis import to_cumulative
from .merge import period_merge

__all__ = [
    "paid_bs_adjustment",
    "reported_bs_adjustment",
    "weight_geometric_decay",
]


def paid_bs_adjustment(
    triangle: Triangle,
    triangle_ult_claim_counts: Triangle,
) -> Triangle:
    """
    Adjust paid losses in `triangle` using the "paid" Berquist-Sherman adjustment for changes in
    claim settlement rate. The adjustment assumes that the latest diagonal of a triangle will
    reflect current claim settlement rates, so paid losses in other cells are adjusted using
    linear interpolation. If the claim disposal rate for a given cell is less than the selected
    disposal rate for that development lag, losses are adjusted upwards (i.e. using the next cell
    for the same experience period); if the claim disposal rate for a given cell is greater than
    the selected disposal rate for that development lag, losses are adjusted downwards (i.e. using
    the previous cell in the same experience period).

    For more details on Berquist-Sherman, see Chapter 13 of this text:
    https://www.casact.org/sites/default/files/database/studynotes_friedland_estimating.pdf

    Args:
        triangle: The Triangle that will be adjusted. It must include `cwp_claims` and `paid_loss`
            as fields.
        triangle_ult_claim_counts: A right edge Triangle that has the ultimate `reported_claims`
            corresponding to each period in `tri`. The mean of `reported_claims` is used in
            disposal rate calculations, so its value can be either an array or a scalar.

    Returns:
        A Triangle with adjusted `paid_loss` values. Note that closed claim counts remain unchanged
        from the original triangle.
    """
    tri_ult_right_edge = triangle_ult_claim_counts.right_edge

    if triangle.is_incremental:
        triangle = to_cumulative(triangle)

    tri_w_disposal_rates = (
        period_merge(
            triangle,
            tri_ult_right_edge,
            suffix="_ult",
        )
        .derive_fields(
            disposal_rate=lambda ob: ob["cwp_claims"]
            / np.mean(ob["reported_claims_ult"]),
        )
        .select(list(triangle.fields) + ["disposal_rate"])
    )

    # We need to get the cell with the latest period for each dev lag. This accounts
    # for the situation when there are multiple dev lags in the latest period (e.g. 0 and 12 months).
    # Otherwise, we can't get a corresponding disposal rate for all dev lags.
    def _get_max_period_cell_for_dev_lag(c: Cell):
        return tri_w_disposal_rates.filter(
            lambda ob: (ob.dev_lag() == c.dev_lag()) and (ob.metadata == c.metadata)
        )[-1]  # relies on sorting of cells by increasing period

    selected_disposal_rates = {
        (ob.dev_lag(), ob.metadata): ob["disposal_rate"]
        for ob in tri_w_disposal_rates.right_edge
    }

    def _get_adjusted_paid_loss(cell: Cell):
        try:
            selected_disposal_rate = selected_disposal_rates[
                (cell.dev_lag(), cell.metadata)
            ]
        except KeyError:
            selected_disposal_rate = _get_max_period_cell_for_dev_lag(cell)[
                "disposal_rate"
            ]
        cell_idx = tri_w_disposal_rates.cells.index(cell)

        if cell["disposal_rate"] == selected_disposal_rate:
            return cell["paid_loss"]
        elif cell["disposal_rate"] > selected_disposal_rate:
            # Settlement is faster than selected, so adjust losses downwards

            # Get cell in same period where dev lag is largest that's < current cell's dev_lag.
            # Defaults to a simple "cell dictionary" with 0 paid loss to handle
            # when cell is leftmost cell in a period.
            prev_cell = (
                tri_w_disposal_rates[cell_idx - 1]
                if (cell_idx > 0)
                and (tri_w_disposal_rates[cell_idx - 1].metadata == cell.metadata)
                else {"paid_loss": 0}
            )

            # The previous disposal rate is 0 if `cell` is the leftmost cell in the period
            if type(prev_cell) is dict:
                prev_selected_disposal_rate = 0
            else:
                try:
                    prev_selected_disposal_rate = selected_disposal_rates[
                        (prev_cell.dev_lag(), prev_cell.metadata)
                    ]
                except KeyError:
                    prev_selected_disposal_rate = _get_max_period_cell_for_dev_lag(
                        prev_cell
                    )["disposal_rate"]

            return np.interp(
                selected_disposal_rate,
                [prev_selected_disposal_rate, cell["disposal_rate"]],
                [prev_cell["paid_loss"], cell["paid_loss"]],
            )
        else:
            # Settlement is slower than selected, so adjust losses upwards

            # Get cell in same period where dev lag is smallest that's > current cell's dev_lag
            next_cell = (
                tri_w_disposal_rates[cell_idx + 1]
                if (cell_idx < (len(tri_w_disposal_rates) - 1))
                and (tri_w_disposal_rates[cell_idx + 1].metadata == cell.metadata)
                else {"paid_loss": 0}
            )

            try:
                next_selected_disposal_rate = selected_disposal_rates[
                    (next_cell.dev_lag(), next_cell.metadata)
                ]
            except KeyError:
                next_selected_disposal_rate = _get_max_period_cell_for_dev_lag(
                    next_cell
                )["disposal_rate"]

            return np.interp(
                selected_disposal_rate,
                [cell["disposal_rate"], next_selected_disposal_rate],
                [cell["paid_loss"], next_cell["paid_loss"]],
            )

    return tri_w_disposal_rates.derive_fields(paid_loss=_get_adjusted_paid_loss)


def reported_bs_adjustment(
    triangle: Triangle,
    annual_severity_trend: float = 0.0,
    sev_trend_method: str | None = None,
):
    """
    Adjust reported losses in `triangle` using "reported" Berquist-Sherman adjustment for changes in case
    reserve adequacy. This method detrends average case outstanding values from the latest diagonal
    using a severity trend.

    NOTE: the current implementation has the option to calculate a severity trend from the data. This
    solution is deterministic and non-ideal. Instead, severity trends should be modelled and provided
    to this method.

    Args:
        triangle: The Triangle for which reported_loss should be adjusted. Note that this triangle must
            include `reported_loss`, `paid_loss`, `open_claims`, and `cwp_claims`.
        annual_severity_trend: The annualized severity trend that will be used to adjust `reported_loss`
            provided as a decimal (e.g. 0.15 is 15% severity trend). Ignored if `sev_trend_method`
            is not None.
        sev_trend_method: Optional method used to calculate severity trend. Options are "latest" or
            "all". If "all", the mean of calculate severity trend for all cells will be used as the
            severity trend; if "latest", instead the mean will be taken only of the latest diagonal
            of `triangle`.

    Returns:
        A triangle with adjusted `reported_loss` values.
    """
    warnings.warn(
        "WARNING: Reported Berquist-Sherman is still under developement, so it may not yet work as intended."
    )

    # Adjustments are made on a calendar period-basis, so the triangle must be
    # regular and not have any experience gaps.
    if not triangle.is_regular() and len(triangle.experience_gaps):
        raise Exception("Severity trends can only be calculated on regular triangles.")

    if triangle.is_incremental:
        triangle = to_cumulative(triangle)

    tri_w_avg_case = triangle.derive_fields(
        average_case_os=lambda ob: (ob["reported_loss"] - ob["paid_loss"])
        / ob["open_claims"],
        average_paid_severity=lambda ob: ob["paid_loss"] / ob["cwp_claims"],
    )

    if sev_trend_method:
        annual_severity_trend = _sev_trend_from_hist_tri(
            tri_w_avg_case, method=sev_trend_method
        )

    def _detrend_average_case_os(cell: Cell):
        # Get the cell in the latest period with the same dev_lag and metadata as the current cell.
        # NOTE this method (in contrast to reduction) relies on triangle.cells being sorted by increasing
        # period so the last cell matching dev_lag and metadata will be the cell with the most
        # recent period.
        diagonal_cell = tri_w_avg_case.right_edge.filter(
            lambda ob: (ob.dev_lag() == cell.dev_lag())
            and (ob.metadata == cell.metadata)
        )[-1]

        trend_period = dev_lag_months(cell.period_end, diagonal_cell.period_end)

        return diagonal_cell["average_case_os"] / (1 + annual_severity_trend) ** (
            trend_period / 12
        )

    return tri_w_avg_case.derive_fields(
        average_case_os=_detrend_average_case_os,
        reported_loss=lambda ob: ob["average_case_os"] * ob["open_claims"]
        + ob["paid_loss"],
    )


def _sev_trend_from_hist_tri(
    triangle: Triangle,
    method: str = "all",  # either "all" or "latest"
):
    if method not in ["all", "latest"]:
        raise Exception(
            f"Unrecognized method `{method}` passed in severity trend calculation."
        )

    def _calculate_sev_trend(cell: Cell):
        # Get cell with the same dev lag in the nearest future period
        # NOTE this method (in contrast to reduction) relies on triangle.cells being sorted by increasing
        # period so the first cell matching dev_lag and metadata will be the cell with the period
        # immediately after the current cell's period.
        future_cells = triangle.filter(
            lambda ob: (ob.dev_lag() == cell.dev_lag()) and (ob.period > cell.period)
        )
        future_cell = (
            future_cells[0] if len(future_cells) else {"average_paid_severity": np.nan}
        )

        trend_period = (
            0
            if type(future_cell) is dict
            else dev_lag_months(cell.period_end, future_cell.period_end)
        )

        return (
            np.nan
            if np.isnan(future_cell["average_paid_severity"])
            else (future_cell["average_paid_severity"] / cell["average_paid_severity"])
            ** (trend_period / 12)
            - 1
        )

    tri_w_sev_trend = triangle.derive_fields(
        severity_trend=_calculate_sev_trend
    ).filter(lambda ob: not np.isnan(ob["severity_trend"]))

    if method == "all":
        return np.mean([ob["severity_trend"] for ob in tri_w_sev_trend])
    elif method == "latest":
        return np.mean([ob["severity_trend"] for ob in tri_w_sev_trend.right_edge])


def weight_geometric_decay(
    triangle: Triangle,
    annual_decay_factor: float,
    basis: str = "evaluation",
    tri_fields: str | list[str] | None = None,
    weight_as_field: bool = True,
) -> Triangle:
    """Implements loss experience period geometric decay weighting. Suppose a triangle's loss experience periods are indexed by t=1:T,
    with T being the latest experience period in the triangle. Then, each loss experience period is weighted by a geometric progression (r)^{(T-t)*frequency},
    where r is the geometric factor and frequency is defined as (resolution of tri) / 12.

    Note that if weight_as_field = False, the processed triangle returned from this function will have fields that are meaningless. The usage
    of this function should be limited towards the preprocessing of triangle data used when fitting loss development models if weight_as_field is False.

    Arguments:
        triangle: triangle input data.
        annual_decay_factor: the geometric factor to apply on an annual basis. Annual rates are
        automatically converted to equal the implied rates that match the resolution of tri. Note that a factor of 1
        (and technically 0) will apply equal weighting to all experience periods, i.e. the triangle will not be modified. Must
        be > 0 and <= 1.
        basis: what basis to weight the triangle on? Available options are `evaluation` (weighting is applied diagonally, i.e. by calendar year)
        and `experience` (weights are applied by column). Default is evaluation period weighting.
        tri_fields: loss fields in tri to geometrically weight. Default is None, in which case
        all fields in the triangle are weighted accordingly.
        weight_as_field: should the weights be returned as an additional field (True) or should the data be weighted in the triangle directly (False)?

    Returns:
        A triangle with a field called geometric_weight (if weight_as_field is True), or a triangle with directly weighted fields.
    """

    # Input checks and setup

    ## Annual decay factor
    if isinstance(annual_decay_factor, float):
        if annual_decay_factor > 1 or annual_decay_factor <= 0:
            raise ValueError("annual_decay_factor must be a scalar > 0 and <= 1")

    ## loss_fields
    if tri_fields is None:
        tri_fields = triangle.fields

    elif isinstance(tri_fields, str):
        if tri_fields not in triangle.fields:
            raise ValueError(f"{tri_fields} not found in tri")
        else:
            tri_fields = [tri_fields]

    elif isinstance(tri_fields, list):
        if not set(tri_fields).issubset(set(triangle.fields)):
            raise ValueError(f"At least one of {tri_fields} not found in `triangle`")

    last_date = (
        triangle.evaluation_dates[-1]
        if basis == "evaluation"
        else triangle.periods[-1][0]
    )

    if weight_as_field:
        triangle_with_weights = triangle.derive_fields(
            geometric_weight=lambda cel: _cell_weight(
                cel, last_date, annual_decay_factor, basis
            )
        )

        return triangle_with_weights
    else:
        decay_weighted_triangle = Triangle(
            [
                _reweight_cell(
                    cell,
                    last_date,
                    annual_decay_factor,
                    fields=tri_fields,
                    weighting_basis=basis,
                )
                for cell in triangle.cells
            ]
        )

        return decay_weighted_triangle


def _cell_weight(cell: Cell, last_date_start, decay_factor, weighting_basis) -> float:
    cell_date = (
        cell.evaluation_date if weighting_basis == "evaluation" else cell.period_start
    )
    months_from_end = month_to_id(last_date_start) - month_to_id(cell_date)
    years_from_end = months_from_end / 12
    return decay_factor**years_from_end


def _reweight_cell(
    cell: Cell,
    last_date_start,
    decay_factor,
    fields: list[str],
    weighting_basis,
) -> float:
    new_weight = _cell_weight(cell, last_date_start, decay_factor, weighting_basis)
    for field in fields:
        cell = cell.derive_fields(**{field: lambda ob: ob[field] * new_weight})
    return cell
