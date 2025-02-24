from datetime import date
from typing import Tuple, Union

import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.interpolate import interp1d

from ..base import Cell, CellValue
from ..date_utils import add_months, eval_date_resolution, period_resolution
from ..errors import TriangleError
from ..triangle import Triangle
from .basis import to_cumulative, to_incremental

__all__ = [
    "disaggregate",
    "disaggregate_development",
    "disaggregate_experience",
]


DEFAULT_INTERPOLATION_FIELDS = [
    "paid_loss",
    "reported_loss",
    "incurred_loss",
    "earned_premium",
]


def disaggregate(
    triangle: Triangle,
    resolution_exp_months: int = 3,
    resolution_dev_months: int = 3,
    fields: list[str] | None = None,
    period_weights: Union[list[float], dict[date, list[float]]] | None = None,
    interpolation_method: str = "linear",
    extrapolate_first_period: bool = True,
    **interpolation_kwargs,
) -> Triangle:
    """
    Disaggregate a triangle to a finer resolution of development and experience periods.
        First it will attempt to disaggregate the triangle across experience periods putting
        different weights to field values over each subperiod. Next it will try to disaggregate
        across development axis applying interpolation.

        Implement this function with caution. It should not be used to refine transaction program
        data. Instead, it can be applied to triangles used for comparison or similar purposes
        during transaction analysis.

    Arguments:
        triangle: The triangle to disaggregate.
        resolution_exp_months: The resolution of experience periods in the output triangle.
        resolution_dev_months: The resolution of evaluation dates in the output triangle.
        fields: The list of fields in the triangle to disaggregate.
        period_weights: Weights for field values applied during disaggregation
            of one experience period. Weights must be from [0, 1], and should sum to 1.
            If set as a dictionary, each experience period will use a separate list of weights.
            If set as a list, the same weights are applied to all experience periods.
            If not supplied, weights are equally distributed across each experience period.
        interpolation_method: The interpolation method to use. See `scipy.interpolate.interp1d`
            to see the list of available interpolation methods.
        extrapolate_first_period: Boolean flag. If true, extrapolation will be applied to get
            field values for development periods prior to the first evaluation date.
        interpolation_kwargs: Keyword arguments passed to the interpolation method.
    Returns:
        A triangle with disaggregated experience periods and evaluation dates
            at the desired resolutions.
    """
    if min(triangle.dev_lags()) < 0:
        raise TriangleError(
            "Cannot disaggregate a triangle with negative development lags."
        )
    disaggregate_experience_triangle = disaggregate_experience(
        triangle,
        resolution_exp_months,
        period_weights,
        fields,
    )
    disaggregate_triangle = disaggregate_development(
        disaggregate_experience_triangle,
        resolution_dev_months,
        fields,
        interpolation_method,
        extrapolate_first_period,
        **interpolation_kwargs,
    )
    return disaggregate_triangle


def disaggregate_development(
    triangle: Triangle,
    resolution_months: int,
    fields: list[str] | None = None,
    interpolation_method: str = "linear",
    extrapolate_first_period: bool = True,
    **interpolation_kwargs,
) -> Triangle:
    """
    Disaggregate a triangle to a finer resolution of development periods
        using a selected interpolation method.

    Arguments:
        triangle: The triangle to disaggregate across development periods.
        resolution_months: The resolution of evaluation dates in the output triangle.
        fields: The list of fields in the triangle to interpolate.
        interpolation_method: The interpolation method to use. See `scipy.interpolate.interp1d`
            to see the list of available interpolation methods.
        extrapolate_first_period: Boolean flag. If true, extrapolation will be applied to get
            field values for development periods prior to the first evaluation date.
        interpolation_kwargs: Keyword arguments passed to the interpolation method.
    Returns:
        A triangle with interpolated evaluation dates at the desired resolution.
    """
    if resolution_months > eval_date_resolution(triangle):
        raise ValueError(
            f"Triangle has finer resolution for evaluation dates than {resolution_months=}."
        )
    elif resolution_months == eval_date_resolution(triangle):
        return triangle
    else:
        if fields is None:
            fields = DEFAULT_INTERPOLATION_FIELDS
        if not any([field in fields for field in triangle.fields]):
            raise ValueError(
                f"list {fields} should include at least one valid field from the input triangle."
            )

        cum_triangle = to_cumulative(triangle) if triangle.is_incremental else triangle
        cells = []
        for _, tri_slice in cum_triangle.slices.items():
            cells += _disaggregate_development_slice(
                tri_slice,
                resolution_months,
                fields,
                interpolation_method,
                extrapolate_first_period,
                **interpolation_kwargs,
            )
        disaggregate_triangle = (
            to_incremental(Triangle(cells))
            if triangle.is_incremental
            else Triangle(cells)
        )
        return disaggregate_triangle


def _disaggregate_development_slice(
    tri_slice: Triangle,
    resolution_months: int,
    fields: list[str],
    interpolation_method: str,
    extrapolate_first_period: bool,
    **interpolation_kwargs,
) -> list[Cell]:
    eval_resolution = eval_date_resolution(tri_slice)

    new_cells = []
    for _, row_cells in tri_slice.slice_period_rows:
        init_cell = row_cells[0]
        row_tri = Triangle(row_cells)

        dev_lags = row_tri.dev_lags(unit="months")
        # need at least 2 cells per period for interpolation
        if len(dev_lags) > 1:
            # check for gaps in development periods
            dev_offset = dev_lags[1] - dev_lags[0]
            for prv, nxt in zip(dev_lags[1:-1], dev_lags[2:]):
                if nxt - prv != dev_offset:
                    raise TriangleError(
                        "Cannot disaggregate a triangle with gaps in development periods."
                    )

            # cells in `row_cells` are already sorted by evaluation_date
            min_lag = init_cell.dev_lag()
            if extrapolate_first_period:
                min_x = np.maximum(0.0, min_lag - eval_resolution + resolution_months)
            else:
                min_x = min_lag
            max_x = row_cells[-1].dev_lag()
            pred_x = np.arange(min_x, max_x + resolution_months, resolution_months)

            field_values = {}
            for field in fields:
                if field not in row_tri.fields:
                    continue

                obs_x = np.array([cell.dev_lag() for cell in row_tri if field in cell])

                num_field_samples = row_tri.select([field]).num_samples
                if num_field_samples == 1:
                    obs_y = np.array([cell[field] for cell in row_tri if field in cell])
                else:
                    obs_y = np.array(
                        [
                            (
                                cell[field]
                                if isinstance(cell[field], np.ndarray)
                                and cell[field].size > 1
                                else np.full((num_field_samples,), cell[field])
                            )
                            for cell in row_tri
                            if field in cell
                        ]
                    ).T

                if extrapolate_first_period:
                    obs_x = np.concatenate([[-resolution_months], obs_x])
                    if num_field_samples == 1:
                        obs_y = np.concatenate([[0], obs_y])
                    else:
                        obs_y = np.concatenate(
                            [np.zeros(num_field_samples).reshape(-1, 1), obs_y], axis=1
                        )

                interp_fn = interp1d(
                    obs_x,
                    obs_y,
                    kind=interpolation_method,
                    fill_value="extrapolate" if extrapolate_first_period else None,
                    **interpolation_kwargs,
                )
                pred_y = interp_fn(pred_x)

                disaggregate_field_values = np.array(
                    [pred_y[..., ndx] for ndx in range(len(pred_x))]
                )
                if (disaggregate_field_values < 0).any():
                    raise ValueError(
                        "Negative field values are generated. Consider changing interpolation method."
                    )
                else:
                    field_values[field] = disaggregate_field_values

            new_cells += [
                init_cell.replace(
                    evaluation_date=add_months(init_cell.period_end, dev_lag),
                    values={field: field_values[field][ndx] for field in field_values},
                )
                for ndx, dev_lag in enumerate(pred_x)
            ]

        # now deal with period containing a single cell
        else:
            if not extrapolate_first_period:
                new_cells += [init_cell]
            else:
                if not new_cells:
                    continue
                cell_lag = init_cell.dev_lag()
                min_x = np.maximum(0.0, cell_lag - eval_resolution + resolution_months)
                pred_x = np.arange(
                    min_x, cell_lag + resolution_months, resolution_months
                )

                last_interpolated_period = sorted(new_cells, key=lambda c: c.period)[
                    -1
                ].period
                relevant_cells = sorted(
                    [
                        cell
                        for cell in new_cells
                        if cell.period == last_interpolated_period
                    ],
                    key=lambda c: c.dev_lag(),
                )[: len(pred_x)]

                new_cells += [
                    init_cell.replace(
                        evaluation_date=add_months(init_cell.period_end, dev_lag),
                        values={
                            field: cell[field]
                            * init_cell[field]
                            / relevant_cells[-1][field]
                            for field in fields
                            if field in cell.values
                        },
                    )
                    for dev_lag, cell in zip(pred_x, relevant_cells)
                ]

    return new_cells


def disaggregate_experience(
    triangle: Triangle,
    resolution_months: int,
    period_weights: Union[list[float], dict[date, list[float]]] | None = None,
    fields: list[str] | None = None,
) -> Triangle:
    """
    Disaggregate a triangle to a finer resolution across experience periods.
        The process allows to put different weights to each subinterval
        within one experience period.

    Arguments:
        triangle: The triangle to disaggregate across experience periods.
        resolution_months: The resolution of experience periods in the output triangle.
        period_weights: Weights for field values applied during disaggregation
            of one experience period. Weights must be from [0, 1], and should sum to 1.
            If set as a dictionary, each experience period will use a separate list of weights.
            If set as a list, the same weights are applied to all experience periods.
            If not supplied, weights are equally distributed across each experience period.
        fields: The list of fields in the triangle to disaggregate.
    Returns:
        A triangle with disaggregated field values at the desired resolution.
    """
    if not triangle.is_semi_regular():
        raise TriangleError("Triangle must be semi-regular.")

    tri_period_resolution = period_resolution(triangle)
    if resolution_months > tri_period_resolution:
        raise ValueError(
            f"Triangle has finer resolution for experience periods than {resolution_months=}."
        )
    elif resolution_months == tri_period_resolution:
        return triangle
    else:
        if fields is None:
            fields = DEFAULT_INTERPOLATION_FIELDS
        if not any([field in fields for field in triangle.fields]):
            raise ValueError(
                f"list {fields} should include at least one valid field from the input triangle."
            )

        if tri_period_resolution % resolution_months != 0:
            raise ValueError(
                f"Triangle experience periods cannot be disaggregated \
                over {resolution_months} months."
            )
        else:
            n_subperiods = tri_period_resolution // resolution_months
            if period_weights is None:
                period_weights = [1 / n_subperiods for i in range(0, n_subperiods)]
            _validate_period_weights(
                period_weights,
                n_subperiods,
                [period[0] for period in triangle.periods],
            )

        cum_triangle = to_cumulative(triangle) if triangle.is_incremental else triangle
        cells = []
        for _, tri_slice in cum_triangle.slices.items():
            cells += _disaggregate_experience_slice(
                tri_slice,
                resolution_months,
                period_weights,
                fields,
            )

        disaggregate_triangle = (
            to_incremental(Triangle(cells))
            if triangle.is_incremental
            else Triangle(cells)
        )
        return disaggregate_triangle


def _disaggregate_experience_slice(
    tri_slice: Triangle,
    resolution_months: int,
    period_weights: Union[list[float], dict[date, list[float]]],
    fields: list[str],
) -> list[Cell]:
    tri_period_resolution = period_resolution(tri_slice)
    n_periods = tri_period_resolution // resolution_months

    new_cells = []
    for cell in tri_slice.cells:
        subperiods = [
            (
                add_months(cell.period_start, k * resolution_months),
                add_months(cell.period_start, (k + 1) * resolution_months)
                - relativedelta(days=1),
            )
            for k in range(0, n_periods)
        ]
        # filter out periods after the cell evaluation date to account for negative dev lags
        subperiods = [
            period for period in subperiods if period[1] <= cell.evaluation_date
        ]
        cell_period_weights = period_weights[: len(subperiods)]
        sublist_total = sum(cell_period_weights)
        cell_period_weights = [weight / sublist_total for weight in cell_period_weights]

        weighted_values = _weight_cell_values(cell, cell_period_weights, subperiods)

        new_cells += [
            Cell(
                period_start=period[0],
                period_end=period[1],
                evaluation_date=cell.evaluation_date,
                metadata=cell.metadata,
                values={
                    field: weighted_values[period][field]
                    for field in cell.values
                    if field in fields
                },
            )
            for period in subperiods
            if period[1] <= cell.evaluation_date
        ]
    return new_cells


def _weight_cell_values(
    cell: Cell,
    period_weights: Union[list[float], dict[date, list[float]]],
    subperiods: list[Tuple[date, date]],
) -> dict[Tuple[date, date], dict[str, CellValue]]:
    period_values = dict()
    for field, value in cell.values.items():
        val = np.array([value]) if np.array(value).ndim == 0 else value
        weights = (
            np.array(period_weights[cell.period_start])
            if isinstance(period_weights, dict)
            else np.array(period_weights)
        )
        new_vals = np.array([v * weights for v in val])
        period_values[field] = {
            p: (v[0] if isinstance(value, (float, int)) else v)
            for p, v in zip(subperiods, new_vals.T)
        }
    # return data reorganized by subperiods
    return {
        key: {
            k: period_values[k][key] for k in period_values if key in period_values[k]
        }
        for key in subperiods
    }


def _validate_period_weights(
    period_weights: Union[list[float], dict[date, list[float]]],
    n_subperiods: int,
    tri_period_start: list[date],
):
    if isinstance(period_weights, list):
        weights = [period_weights]
    elif isinstance(period_weights, dict):
        weights = list(period_weights.values())
        if not all(ps in period_weights.keys() for ps in tri_period_start):
            raise ValueError(
                f"Must supply weights for all periods in {tri_period_start}."
            )
    else:
        raise TypeError(f"Set {period_weights=} as a list or dictionary.")

    if not all(len(w) == n_subperiods for w in weights):
        raise ValueError(
            f"All lists in variable {period_weights=} should have length {n_subperiods}."
        )
    if not all(0 <= val <= 1 for w in weights for val in w):
        raise ValueError(
            f"Each weight in {period_weights} must be in the range [0, 1]."
        )
    if not all(sum(w) == 1 for w in weights):
        raise ValueError(f"Weights in {period_weights} should sum to 1.")
    return
