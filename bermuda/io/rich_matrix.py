import numpy as np

from ..base import CumulativeCell, IncrementalCell
from ..date_utils import add_months, id_to_month, is_triangle_monthly, resolution_delta
from ..matrix import (
    DisaggregatedPredictedValue,
    DisaggregatedValue,
    Matrix,
    MatrixIndex,
    MissingValue,
    PredictedValue,
)
from ..triangle import Triangle

__all__ = [
    "triangle_to_rich_matrix",
    "rich_matrix_to_triangle",
]


def triangle_to_rich_matrix(
    tri: Triangle,
    eval_resolution: int | None = None,
    fields: list[str] | None = None,
) -> Matrix:
    """Convert a triangle to a rich matrix format.

    The rich matrix format is similar to the matrix format, but it somewhat more flexible. It
    can handle missing data, data that needs to be disaggregated, and predicted cells.
    """
    # Ensure that the triangle meets basic data standards
    if not is_triangle_monthly(tri):
        raise ValueError(
            "Triangle cannot be converted to matrices.  The `period_start` of each triangle "
            "cell must be on the first day of the month.  Also, the `period_end` and "
            "`evaluation_date`s of each triangle cell must be on the last day of the month.  "
            "One or more of these conditions is not satisfied for %s" % tri
        )

    # Build the index
    index = MatrixIndex.from_triangle(
        tri, eval_resolution=eval_resolution, fields=fields
    )
    next_disagg_id = 0

    # Figure out maximum dimensions
    if fields is None:
        if not tri.fields:
            raise Exception("Must specify fields if triangle has none")
        fields = tri.fields

    _, _, max_period, max_dev = index.resolve_indices(
        tri.metadata[0], fields[0], tri.periods[-1][0], tri.dev_lags()[-1]
    )
    data_dims = (len(tri.metadata), len(fields), max_period + 1, max_dev + 1)
    covered_dims = (len(tri.metadata), max_period + 1, max_dev + 1)

    # Fill in the data
    data = np.full(data_dims, None, dtype=object)
    is_covered = np.full(covered_dims, False)
    for cell in tri:
        for field, raw_value in cell.values.items():
            if field in fields:
                # Grab indices for the cell
                meta_ndx, field_ndx, exp_start_ndx, dev_ndx = index.resolve_indices(
                    cell.metadata, field, cell.period_start, cell.dev_lag()
                )
                _, _, exp_end_ndx, _ = index.resolve_indices(
                    cell.metadata, field, cell.period_end, cell.dev_lag()
                )

                # Mark the cells as covered
                for i, exp_ndx in enumerate(range(exp_end_ndx, exp_start_ndx - 1, -1)):
                    is_covered[meta_ndx, exp_ndx, dev_ndx + i] = True
                # Fill in the matrix with values
                if isinstance(raw_value, np.ndarray):
                    value = (
                        float(raw_value)
                        if raw_value.size == 1
                        else PredictedValue(raw_value)
                    )
                else:
                    value = raw_value
                if exp_start_ndx == exp_end_ndx:
                    data[meta_ndx, field_ndx, exp_start_ndx, dev_ndx] = value
                else:
                    disagg_value = (
                        DisaggregatedPredictedValue(
                            id=next_disagg_id, value=value.value
                        )
                        if isinstance(value, PredictedValue)
                        else DisaggregatedValue(id=next_disagg_id, value=value)
                    )
                    next_disagg_id += 1
                    for i, exp_ndx in enumerate(
                        range(exp_end_ndx, exp_start_ndx - 1, -1)
                    ):
                        data[meta_ndx, field_ndx, exp_ndx, dev_ndx + i] = disagg_value

    # Fill in missing values that are covered by other cells
    next_missing_id = 0
    for meta_ndx in range(data.shape[0]):
        for exp_ndx in range(data.shape[2]):
            for dev_ndx in range(data.shape[3]):
                if is_covered[meta_ndx, exp_ndx, dev_ndx]:
                    for field_ndx in range(data.shape[1]):
                        if data[meta_ndx, field_ndx, exp_ndx, dev_ndx] is None:
                            data[meta_ndx, field_ndx, exp_ndx, dev_ndx] = MissingValue(
                                next_missing_id
                            )
                            next_missing_id += 1

    return Matrix(data=data, index=index, incremental=tri.is_incremental)


def rich_matrix_to_triangle(mat: Matrix) -> Triangle:
    period_starts = [
        id_to_month(mat.index._exp_origin + i * mat.index._exp_resolution)
        for i in range(mat.data.shape[2])
    ]
    period_ends = [
        id_to_month(
            mat.index._exp_origin + (i + 1) * mat.index._exp_resolution - 1, False
        )
        for i in range(mat.data.shape[2])
    ]
    min_resolution = min(mat.index._exp_resolution, mat.index._dev_resolution)
    dev_lags = [
        float(mat.index._dev_origin + i * min_resolution)
        for i in range(mat.data.shape[3])
    ]

    cells = []

    for i, meta in enumerate(mat.index._slices):
        for j, (period_start, period_end) in enumerate(zip(period_starts, period_ends)):
            for k, dev_lag in enumerate(dev_lags):
                raw_values = mat.data[i, :, j, k]
                values = {
                    name: (val.value if isinstance(val, PredictedValue) else val)
                    for name, val in zip(mat.index._fields, raw_values)
                    if val is not None
                    and not isinstance(
                        val,
                        (MissingValue, DisaggregatedValue, DisaggregatedPredictedValue),
                    )
                }
                if values:
                    if not mat.incremental:
                        new_cell = CumulativeCell(
                            period_start=period_start,
                            period_end=period_end,
                            evaluation_date=add_months(period_end, dev_lag),
                            metadata=meta,
                            values=values,
                        )
                    else:
                        prev_eval = (
                            resolution_delta(period_start, (-1, "days"))
                            if not k
                            else add_months(period_end, dev_lags[k - 1])
                        )
                        new_cell = IncrementalCell(
                            period_start=period_start,
                            period_end=period_end,
                            evaluation_date=add_months(period_end, dev_lag),
                            prev_evaluation_date=prev_eval,
                            metadata=meta,
                            values=values,
                        )
                    cells.append(new_cell)

    return Triangle(cells)
