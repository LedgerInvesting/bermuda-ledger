from dataclasses import replace
from typing import Any, Callable, Literal, Optional, get_args

import numpy as np
import toolz as tlz

from ..base import Cell, CellValue, CumulativeCell, IncrementalCell, Metadata
from ..errors import TriangleError
from ..triangle import Triangle

BLEND_METHOD_TYPE = Literal["mixture", "linear"]

__all__ = [
    "blend",
    "blend_samples",
    "blend_cells",
    "summarize",
    "summarize_cell_values",
    "split",
]


NON_LOSS_METRICS = {
    "earned_premium",
    "used_earned_premium",
    "earned_exposure",
    "written_premium",
    "written_exposure",
    "implied_atu",
    "bf_weight",
    "geometric_weight",
}

SUMMARIZE_DEFAULTS = {
    "paid_loss": lambda vd: _conforming_sum(vd["paid_loss"]),
    "reported_loss": lambda vd: _conforming_sum(vd["reported_loss"]),
    "incurred_loss": lambda vd: _conforming_sum(vd["incurred_loss"]),
    "reported_claims": lambda vd: _conforming_sum(vd["reported_claims"]),
    "open_claims": lambda vd: _conforming_sum(vd["open_claims"]),
    "closed_claims": lambda vd: _conforming_sum(vd["closed_claims"]),
    "closed_with_pay_claims": lambda vd: _conforming_sum(vd["closed_with_pay_claims"]),
    "reported_count": lambda vd: _conforming_sum(vd["reported_count"]),
    "open_count": lambda vd: _conforming_sum(vd["open_count"]),
    "closed_count": lambda vd: _conforming_sum(vd["closed_count"]),
    "closed_with_pay_count": lambda vd: _conforming_sum(vd["closed_with_pay_count"]),
    "earned_premium": lambda vd: _conforming_sum(vd["earned_premium"]),
    "used_earned_premium": lambda vd: _conforming_sum(vd["used_earned_premium"]),
    "earned_exposure": lambda vd: _conforming_sum(vd["earned_exposure"]),
    "written_premium": lambda vd: _conforming_sum(vd["written_premium"]),
    "written_exposure": lambda vd: _conforming_sum(vd["written_exposure"]),
    "implied_atu": lambda vd: _conforming_weighted_average(
        vd["implied_atu"], vd["reported_loss"]
    ),
    "bf_weight": lambda vd: _conforming_weighted_average(
        vd["bf_weight"], vd["reported_loss"]
    ),
    "geometric_weight": lambda vd: _conforming_weighted_average(
        vd["geometric_weight"], vd["reported_loss"]
    ),
    "log_industry_lr": lambda vd: _conforming_weighted_average(
        np.exp(vd["log_industry_lr"]), vd["earned_premium"], np.log
    ),
}

# For Bornhuetter-Ferguson method weighting
SUMMARIZE_DEFAULTS |= {
    f"{flavor}_loss_developed": lambda vd: _conforming_sum(
        vd[f"{flavor}_loss_developed"]
    )
    for flavor in ["incurred", "paid", "reported"]
}

SUMMARIZE_DEFAULTS |= {
    f"{flavor}_loss_prior": lambda vd: _conforming_sum(vd[f"{flavor}_loss_prior"])
    for flavor in ["incurred", "paid", "reported"]
}


# noinspection PyDefaultArgument
def summarize_cell_values(
    cells: list[Cell],
    agg_fns: Optional[dict[str, Callable]] = None,
    summarize_premium: bool = True,
) -> dict[str, CellValue]:
    """Combine all of the values in `cells` into a single dictionary.

    The semantics of combining each individual field in `values` is controlled by the
    `agg_fns` argument. Each summarization operation is passed a dictionary of all cell values
    so that fields whose aggregation depends on multiple values (i.e., loss ratio) can work
    correctly.

    If a key is present in some, but not all cells, default behavior is to pass in explicit
    `None` s for the missing keys into the aggregation function. Each aggregation function has the
    flexibility to decide whether to return a value in the presence of `None` s.
    """
    agg_fns = SUMMARIZE_DEFAULTS if agg_fns is None else SUMMARIZE_DEFAULTS | agg_fns

    value_keys = set([k for cell in cells for k in cell.values.keys()])
    for key in value_keys:
        if key.lower() not in agg_fns:
            raise TriangleError(f"Don't know how to aggregate `{key}` values")
    loss_keys = value_keys - NON_LOSS_METRICS
    non_loss_keys = value_keys & NON_LOSS_METRICS
    cell_raw_values = {
        key: [cell.values.get(key, None) for cell in cells] for key in value_keys
    }
    if summarize_premium:
        values_summaries = {
            key: agg_fns[key.lower()](cell_raw_values) for key in value_keys
        }
        return values_summaries

    else:
        non_loss_indices = _non_loss_distinct_indices(cells)
        cell_non_loss_values = {
            key: [elem for idx, elem in enumerate(values) if idx in non_loss_indices]
            for key, values in cell_raw_values.items()
        }

        loss_summaries = {
            key: agg_fns[key.lower()](cell_raw_values) for key in loss_keys
        }
        non_loss_summaries = {
            key: cell_non_loss_values[key][0] for key in non_loss_keys
        }

        return {
            **loss_summaries,
            **non_loss_summaries,
        }


def _non_loss_distinct_indices(cells: list[Cell]) -> list[int]:
    # per_occurrence_limit should work as a loss detail, we will keep the max limit
    all_limits = [
        cell.metadata.per_occurrence_limit
        for cell in cells
        if cell.metadata.per_occurrence_limit is not None
    ]
    max_limit = max(all_limits) if all_limits else None
    generic_metadata = [
        replace(cell.metadata, loss_details={}, per_occurrence_limit=max_limit)
        for cell in cells
    ]
    distinct_metadata = set()
    distinct_idxs = []
    for idx, meta in enumerate(generic_metadata):
        if meta not in distinct_metadata:
            distinct_metadata |= {meta}
            distinct_idxs.append(idx)
    return distinct_idxs


def _conforming_sum(values: list[CellValue]) -> CellValue:
    """Compute the conforming sum of a list of values."""
    total: CellValue = 0
    for val in values:
        if val is None:
            continue
        # If we're summing two arrays, check shape
        if not np.isscalar(total) and not np.isscalar(val) and total.shape != val.shape:
            raise ValueError(
                f"Cannot sum arrays with shapes {total.shape} and {val.shape}"
            )
        total += val
    return total


def _conforming_weighted_average(
    values: list[CellValue],
    weights: list[CellValue],
    post_transform: Callable = lambda cell: cell,
) -> CellValue:
    """Compute the conforming weighted average of a list of values."""
    total: CellValue = 0
    for val, weight in zip(values, weights):
        if val is None:
            continue
        # If we're averaging two arrays, check shape
        if not np.isscalar(total) and not np.isscalar(val) and total.shape != val.shape:
            raise ValueError(
                f"Cannot average arrays with shapes {total.shape} and {val.shape}"
            )
        total += val * weight
    return post_transform(
        total / sum(weight for weight in weights if weight is not None)
    )


def blend_samples(
    values: list[CellValue],
    weights: list[float] = None,
    method: BLEND_METHOD_TYPE = "mixture",
    seed: int = None,
) -> CellValue:
    if weights is None:
        weights = np.repeat(1 / len(values), len(values))
    else:
        weights = np.asarray(weights)

    if len(weights) != len(values):
        raise ValueError(
            "The number of weights does not match the number of cell values"
        )

    if method == "mixture":
        return _mixture_blend(values, weights, seed)
    elif method == "linear":
        return _linear_blend(values, weights)
    else:
        raise ValueError(f"Unexpected blending method `{method}`.")


def _linear_blend(values: list[CellValue], weights: list = None):
    """Combine cell values by linear blending."""

    values = [[v] if isinstance(v, (float, int)) else v for v in values]
    S = max(len(v) for v in values)
    M = len(values)

    if np.ndim(weights) == 1:
        weights = np.array(weights).reshape((M, 1))

    # Create a 2-dimensional array of values ready for matrix multiplication
    matched_values = np.empty((S, M))
    for idx, val in enumerate(values):
        if len(val) != S and len(val) != 1:
            raise ValueError(
                f"Linear blending requires values of length 1 or {S}, not {len(val)}."
            )
        if len(val) == 1:
            matched_values[:, idx] = np.tile(val, S)
        else:
            matched_values[:, idx] = val

    blend = matched_values @ weights
    return blend.reshape((S,))


def _mixture_blend(
    values: list[CellValue], weights: list = None, seed: int = None
) -> CellValue:
    """Combine a set of samples of size M, each containing an S x 1 vector of samples,
    by a set of weights."""

    if round(sum(weights), 6) != 1:
        raise ValueError(f"Sum of weight parameters in {weights} must be exactly 1")

    np.random.seed(seed)

    M = len(values)
    S = np.shape(values[0])[0]

    blend_idx = np.random.choice(range(M), S, p=weights)
    new_cell_values = np.empty((S,))
    for idx, v in enumerate(values):
        new_cell_values[blend_idx == idx] = v[blend_idx == idx]

    return new_cell_values


def blend_cells(
    cells: list[Cell],
    weights: list[float],
    method: BLEND_METHOD_TYPE,
    seed: Optional[int],
) -> Cell:
    location_str = (
        f"period {cells[0].period}, evaluation date {cells[0].evaluation_date}"
    )

    # Ensure that all cells have the same set of fields
    fields = set(cells[0].values.keys())
    for cell in cells[1:]:
        if set(cell.values.keys()) != fields:
            raise ValueError("Cannot blend cells with an inconsistent set of fields")

    # Blend each field in values, one by one
    clean_values = {}
    for field in fields:
        field_vals = [cell[field] for cell in cells]

        # check that all values have the same class for mixture blending
        if method == "mixture" and not all(
            [isinstance(val, type(field_vals[0])) for val in field_vals[1:]]
        ):
            raise TypeError(
                f"Attempted to mixture blend triangle with inconsistent field types at {location_str}"
            )

        # Check that scalar values are all the same for mixture blending
        # Linear blending can blend distinct scalars
        if method == "mixture" and np.isscalar(field_vals[0]):
            if any([val != field_vals[0] for val in field_vals[1:]]):
                raise ValueError(
                    f"Attempted to mixture blend triangle with inconsistent scalar values at {location_str}"
                )
            clean_values[field] = field_vals[0]
        # otherwise, do method blending
        else:
            clean_values[field] = blend_samples(field_vals, weights, method, seed)

    return cells[0].replace(values=clean_values)


def blend(
    triangles: list[Triangle],
    weights: list[float] | dict[str, np.ndarray] | None = None,
    method: BLEND_METHOD_TYPE = "mixture",
    seed: int | None = None,
) -> Triangle:
    """Return a weighted blend of triangles by their values.

    There are two distinct methods of blending.
    Mixture blending samples cell values in
    proportion to their weight. Linear blending
    is a linear weighted average of the cell values.
    Mixture blending only applies to distributions of
    cell values with the same dimensions, but linear
    blending can be applied between distributions
    and scalars alike.

    Args:
        triangles: The list of triangles to blend.
        weights: The optional list or dictionary of weights.
            Cell-wise weights are passed as a dictionary.
        method: The blending method to use. Can be one of
            ['mixture', 'linear']. Defaults to `mixture`.
        seed: The seed to use for mixture blending.

    Returns:
        A blended triangle.
    """

    if not isinstance(triangles, list):
        raise TypeError(
            f"Can only blend a list of triangles, not a type {type(triangles)}."
        )

    if len(triangles) <= 1 and weights[0] != 1.0:
        raise ValueError(
            f"Blending single triangles requires weight = [1.0] not {weights}."
        )

    if method.lower() not in get_args(BLEND_METHOD_TYPE):
        raise ValueError(
            f"Blending method `{method}` not found. Available methods are {get_args(BLEND_METHOD_TYPE)}.`"
        )
    else:
        method = method.lower()

    # Make sure that all of the triangles are the same length and have the same cell type
    # We'll check to make sure that all of the cell indices and metadata match later
    n_cells = len(triangles[0])
    if any([len(tri) != n_cells for tri in triangles[1:]]):
        raise ValueError("All triangles in blend must have the same number of elements")
    if any(
        [type(tri.cells[0]) != type(triangles[0].cells[0]) for tri in triangles[1:]]  # noqa: E721
    ):
        raise ValueError("All triangles in blend must have the same cell type")

    if triangles[0].is_incremental:
        index_triangles = [
            {
                (
                    cell.metadata,
                    cell.period,
                    cell.evaluation_date,
                    cell.prev_evaluation_date,
                ): cell
                for cell in tri
            }
            for tri in triangles
        ]
    else:
        index_triangles = [
            {(cell.metadata, cell.period, cell.evaluation_date): cell for cell in tri}
            for tri in triangles
        ]

    # re-structure weights dependent on input type
    if weights is None:
        weight_list = [None] * n_cells
    elif isinstance(weights, dict):
        weights_2d = {k: np.atleast_2d(v) for k, v in weights.items()}
        weight_list = [weight for weight in np.concatenate(list(weights_2d.values())).T]
        if len(weight_list) != 1 and len(weight_list) != n_cells:
            raise ValueError(
                "Dimensions of `weights` do not match those of `triangles`. Either a single "
                "set of weights should be supplied that will be applied to all cells "
                "in `triangles`, or exactly one set of weights for each cell in `triangles` "
                "should be supplied for cell-wise blending."
            )
        if len(weight_list) == 1:
            weight_list = weight_list * n_cells
    elif isinstance(weights, list):
        weight_list = [weights] * n_cells
    else:
        raise TypeError(
            f"`weights` passed to bermuda.utils.blend has unknown type {type(weights)}."
        )

    blended_cells = []
    for ndx, cell_weights in zip(index_triangles[0], weight_list):
        try:
            ndx_cells = [ndx_tri[ndx] for ndx_tri in index_triangles]
        except KeyError:
            raise ValueError(
                "All triangles in a blend must have an identical set of coordinates"
            )
        blended_cells.append(blend_cells(ndx_cells, cell_weights, method, seed))

    return Triangle(blended_cells)


def split(triangle: Triangle, detail_keys: list[str]) -> dict[Any, Triangle]:
    """Turn a Triangle into a dictionary of Triangles grouped by specified detail keys."""

    def key_fn(cell):
        detail_vals = [cell.metadata.details.get(key, None) for key in detail_keys]
        return tuple(detail_vals)

    grouped_cells = tlz.groupby(key_fn, triangle.cells)
    return tlz.valmap(Triangle, grouped_cells)


def summarize(
    triangle: Triangle,
    summary_fns: Optional[dict[str, Callable]] = None,
    summarize_premium: bool = True,
) -> Triangle:
    """Aggregate a Triangle across metadata.

    For example, given a triangle with state and coverage-level detail, this would generate a
    triangle where all cells with the same coordinates are merged, and any metadata that varies
    over the triangle would be dropped.

    Args:
        triangle: The single triangle to summarize.
        summary_fns: A dictionary of functions to use to summarize each field in the triangle. If
            `None`, a default set of functions is supplied. If specified, the dictionary is added to the default set of summary functions, which makes it possible to easily add functions for custom fields without affecting defaults.
        summarize_premium: Whether to summarize the premium field. If False, the premium field
            will not be summed accross cells, but will be copied from the first matching cell
            in the triangle.

    Returns:
        A summarized triangle with a single cell for each unique combination of coordinates.
    """
    # noinspection PyShadowingNames
    common_metadata = _metadata_gcd(triangle)

    summary_cells = []
    if triangle.is_incremental:
        grouped_cells = tlz.groupby(
            lambda cell: (cell.period, cell.evaluation_date, cell.prev_evaluation_date),
            triangle.cells,
        )
        for (period, eval_date, prev_eval_date), cells in grouped_cells.items():
            summary_cells.append(
                IncrementalCell(
                    period_start=period[0],
                    period_end=period[1],
                    evaluation_date=eval_date,
                    prev_evaluation_date=prev_eval_date,
                    metadata=common_metadata,
                    values=summarize_cell_values(cells, summary_fns),
                )
            )
    else:
        grouped_cells = tlz.groupby(
            lambda cell: (cell.period, cell.evaluation_date), triangle.cells
        )
        for (period, eval_date), cells in grouped_cells.items():
            summary_cells.append(
                CumulativeCell(
                    period_start=period[0],
                    period_end=period[1],
                    evaluation_date=eval_date,
                    metadata=common_metadata,
                    values=summarize_cell_values(cells, summary_fns, summarize_premium),
                )
            )

    return Triangle(summary_cells)


def _metadata_gcd(triangle: Triangle[Cell]) -> Metadata:
    if not triangle.has_consistent_risk_basis:
        raise TriangleError("All cells in `summarize` must have the same risk basis")
    if not triangle.has_consistent_currency:
        raise TriangleError("All cells in `summarize` must have the same currency")

    return Metadata(
        risk_basis=triangle.cells[0].risk_basis,
        currency=triangle.cells[0].currency,
        country=_metadata_attr_gcd(triangle, "country"),
        per_occurrence_limit=_metadata_attr_gcd(triangle, "per_occurrence_limit"),
        loss_definition=_metadata_attr_gcd(triangle, "loss_definition"),
        reinsurance_basis=_metadata_attr_gcd(triangle, "reinsurance_basis"),
        details=_details_gcd([cell.metadata.details for cell in triangle.cells]),
        loss_details=_details_gcd(
            [cell.metadata.loss_details for cell in triangle.cells]
        ),
    )


def _details_gcd(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    # Find the keys that are present in all of the dicts
    common_keys = set(dicts[0].keys())
    for dct in dicts[1:]:
        common_keys &= set(dct.keys())

    # Check if those keys have the same values in all of the dicts
    common_values = {}
    for key in common_keys:
        first_value = dicts[0][key]
        consistent_value = True
        for dct in dicts[1:]:
            if dct[key] != first_value:
                consistent_value = False
                break
        if consistent_value and first_value is not None:
            common_values[key] = first_value

    return common_values


def _metadata_attr_gcd(triangle: Triangle[Cell], attr_name: str):
    first_val = getattr(triangle.cells[0].metadata, attr_name)
    if all([getattr(cell.metadata, attr_name) == first_val for cell in triangle]):
        return first_val
    return None
