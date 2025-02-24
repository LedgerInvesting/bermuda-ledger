import numpy as np

from ..base import CumulativeCell, IncrementalCell
from ..date_utils import add_months, id_to_month, is_triangle_monthly, resolution_delta
from ..matrix import Matrix, MatrixIndex
from ..triangle import Triangle

__all__ = [
    "triangle_to_matrix",
    "matrix_to_triangle",
]


def triangle_to_matrix(
    tri: Triangle,
    eval_resolution: int | None = None,
    fields: list[str] | None = None,
) -> Matrix:
    """Convert a triangle to a matrix format.

    The matrix format is well-suited for clean data with consistent experience period lengths
    and evaluation date deltas. If a triangle is not regular or if it has Predictions, it
    cannot be converted to a matrix.
    """
    # Ensure that the triangle meets basic data standards
    if not is_triangle_monthly(tri):
        raise Exception("Only monthly Triangles can be converted to matrices")
    if not tri.is_semi_regular():
        raise Exception("Only semi-regular Triangles can be converted to matrices")

    if fields is None:
        if not tri.fields:
            raise Exception("Must specify fields if triangle has none")
        fields = tri.fields

    # Build the index
    index = MatrixIndex.from_triangle(
        tri, eval_resolution=eval_resolution, fields=fields
    )

    # Figure out maximum dimensions
    _, _, max_period, max_dev = index.resolve_indices(
        tri.metadata[0], fields[0], tri.periods[-1][0], tri.dev_lags()[-1]
    )
    data_dims = (len(tri.metadata), len(fields), max_period + 1, max_dev + 1)

    # Fill in the data
    data = np.full(data_dims, np.nan)
    for cell in tri:
        for field, value in cell.values.items():
            if field in fields:
                data[
                    index.resolve_indices(
                        cell.metadata, field, cell.period_start, cell.dev_lag()
                    )
                ] = float(value)

    return Matrix(data=data, index=index, incremental=tri.is_incremental)


def matrix_to_triangle(mat: Matrix) -> Triangle:
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
    dev_lags = [
        float(mat.index._dev_origin + i * mat.index._dev_resolution)
        for i in range(mat.data.shape[3])
    ]

    cells = []

    for i, meta in enumerate(mat.index._slices):
        for j, (period_start, period_end) in enumerate(zip(period_starts, period_ends)):
            for k, dev_lag in enumerate(dev_lags):
                raw_values = mat.data[i, :, j, k]
                values = {
                    name: val
                    for name, val in zip(mat.index._fields, raw_values)
                    if not np.isnan(val)
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
