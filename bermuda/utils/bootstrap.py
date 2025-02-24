import numpy as np
import scipy.stats as stat

from ..date_utils import eval_date_resolution
from ..triangle import Triangle


def bootstrap(
    triangle: Triangle,
    n: int,
    seed: int,
    field: str | list[str] | None = None,
) -> list[Triangle]:
    """Sample bootstrap replicates from a triangle.

    Traditional bootstrapping would sample
    triangle cells with replacement, but this is
    inappropriate because cells are not iid
    realizations. This function implements
    two different types of bootstrapping
    that preserve structure between cells:

        * If a triangle has multiple development
          lags and experience periods, the 'ata'
          method is implemented, which resamples
          volume-weighted age-to-age factors
          between experience periods.
        * If the triangle has a single experience
          period, diagonal, or development
          lag, then maximum entropy bootstrapping
          is used. Maximum entropy bootstrapping (Vinod, 2006)
          is a bootstrapping method for time series
          data that does not require stationarity,
          detrending, or block bootstrapping.

    The function returns a multi-slice triangle
    with metadata details key-value pair ('bootstrap', i),
    where i is the i-th replicate.

    Arguments:
        triangle: The triangle to re-sample.
        n: The number of boostrap replicates to sample.
        seed: The RNG seed.
        field: The field, or list of fields, to resample
            in the triangle. Defaults to `None`, which
            selects all fields.

    Returns:
        A list of (potentially multi-slice) triangles, one for
        each bootstrap replicate.
    """

    if n <= 0:
        raise ValueError("Argument `n` to `bootstrap` must be a positive integer.")
    boots = (
        _bootstrap_slice(triangle_slice, n, seed, field)
        for _, triangle_slice in triangle.slices.items()
    )
    zipped = zip(*boots)
    return [sum(boot) for boot in zipped]


def _bootstrap_slice(
    triangle: Triangle,
    n: int,
    seed: int,
    field: str | list[str] | None = None,
) -> Triangle:
    """Bootstrap a single triangle slice.

    Returns:
        A multi-slice triangle, one slice per bootstrap replicate.
    """
    rng = np.random.default_rng(seed)
    fields = (
        [field]
        if isinstance(field, str)
        else field
        if isinstance(field, list)
        else triangle.fields
    )
    if (
        len(triangle.dev_lags()) > 1
        and len(triangle.periods) > 1
        and len(triangle.evaluation_dates) > 1
    ):
        method = "atas"
    else:
        method = "maximum_entropy"
    if method == "atas":
        atas, volume_weights = _empirical_atas(triangle, fields)
        S = {
            lag: {field: len(v) for field, v in ata.items()}
            for lag, ata in atas.items()
        }
        weights = {
            i: {
                lag: {field: volume_weights[lag][field] for field, s in count.items()}
                for lag, count in S.items()
            }
            for i in range(n)
        }
        resampled_atas = {
            i: {
                lag: {
                    field: atas[lag][field][
                        rng.choice(range(len(p)), size=len(p), p=p, replace=True)
                    ]
                    for field, p in w.items()
                }
                for lag, w in lag_atas.items()
            }
            for i, lag_atas in weights.items()
        }
        replicates = (
            _develop_triangle_by_atas(triangle, resampled_atas[i]).derive_metadata(
                details=lambda cell: {**cell.metadata.details, **{"bootstrap": i}}
            )
            for i in range(n)
        )
    elif method == "maximum_entropy":
        replicates = []
        for i in range(n):
            meboot = {
                field: maximum_entropy_ensemble(
                    [cell[field] for cell in triangle],
                    rng.uniform(size=len(triangle)),
                    L=(0, max(cell[field] for cell in triangle)),
                )
                for field in fields
            }
            cells = []
            for cell_idx, cell in enumerate(triangle):
                cells.append(
                    cell.derive_fields(
                        **{field: values[cell_idx] for field, values in meboot.items()}
                    )
                )
            replicates += [
                Triangle(cells).derive_metadata(
                    details=lambda cell: {**cell.metadata.details, "bootstrap": i}
                )
            ]
    return replicates


def _develop_triangle_by_atas(triangle, resampled_atas) -> Triangle:
    cells = []
    period_lookup = {period: i for i, period in enumerate(triangle.periods)}
    initial_values = {
        period: min(triangle.filter(lambda ob: ob.period == period).dev_lags())
        for period in period_lookup
    }
    for cell in triangle:
        if cell.dev_lag() == initial_values[cell.period]:
            cells.append(cell)
            values = cells[-1].values
        else:
            period_idx = period_lookup[cell.period]
            values = {
                **cell.values,
                **{
                    field: (
                        v * resampled_atas[cell.dev_lag()][field][period_idx]
                        if cell.values.get(field)
                        else None
                    )
                    for field, v in values.items()
                },
            }
            cells.append(
                cell.replace(
                    values=values,
                )
            )
    return Triangle(cells)


def _empirical_atas(triangle: Triangle, fields: list[str]):
    atas = {}
    zipped_lags = zip(triangle.dev_lags()[1:], triangle.dev_lags()[:-1])
    resolution = eval_date_resolution(triangle)

    def _safe_ata_division(x, y):
        safe_x = 1 if x is None or not x else x
        safe_y = 1 if y is None or not y else y
        return safe_x / safe_y

    for lag, prev_lag in zipped_lags:
        clipped_tri = triangle.clip(min_dev=prev_lag, max_dev=lag)
        lag_cells = [
            (next_cell, prev_cell)
            for next_cell, prev_cell in zip(clipped_tri[1:], clipped_tri[:-1])
            if next_cell.period == prev_cell.period
        ]
        atas[lag] = {
            field: np.array(
                [
                    _safe_ata_division(
                        next_cell.values.get(field), prev_cell.values.get(field)
                    )
                    for next_cell, prev_cell in lag_cells
                ]
            )
            for field in fields
        }
    volume_weight = {
        lag: {
            field: _normalize(
                np.array(
                    [
                        cell[field]
                        for cell in triangle.filter(
                            lambda cell: cell.dev_lag() == lag - resolution
                        )
                    ]
                )[: len(vals)]
            )
            for period in triangle.periods
            for field, vals in field_vals.items()
        }
        for lag, field_vals in atas.items()
    }
    return atas, volume_weight


def _normalize(x: np.array):
    if sum(x) == 0:
        return np.repeat(1 / x.size, x.size)
    return x / x.sum()


def maximum_entropy_ensemble(
    x: list[float], U: list[float], L: tuple[float, float] | None = None
) -> np.ndarray:
    """Applies the maximum entropy bootstrap algorithm of Vinod (2006).

    See the following links for Python and R implementations
    that this function is heavily based on. However, note that the
    Python implementations are wrong in that they don't appear to
    sort the final quantiles:
        * https://gist.github.com/TomKealy/975a7f639dec4febad5964950023a6c0
        * https://github.com/kirajcg/pymeboot/blob/master/pymeboot/meboot.py
        * https://github.com/cran/meboot/blob/master/R/meboot.R

    - Vinod, H.D. 2006. Maximum entropy ensembles for time series inference
        in economics. Journal of Asian Economics. 17 (6), 955-978.
    """
    LIMITS = 0.1
    if len(x) == 1:
        return x
    if all(x[0] == i for i in x[1:]):
        return x
    if any(i is None for i in x):
        raise ValueError("Cannot bootstrap missing values using maximum entropy.")
    if any(0 > u > 1 for u in U):
        raise ValueError(
            "U should be a list of draws from the Uniform(0, 1) distribution."
        )

    sorted_x, indices = map(np.array, zip(*sorted([v, i] for i, v in enumerate(x))))
    if L is None:
        trimmed_mean = stat.trim_mean(abs(np.diff(x)), LIMITS)
    z_t = [
        L[0] if L else sorted_x[0] - trimmed_mean,
        *(sorted_x[:-1] + sorted_x[1:]) / 2,
        L[1] if L else sorted_x[-1] + trimmed_mean,
    ]
    desired_means = [
        0.75 * sorted_x[0] + 0.25 * sorted_x[1],
        *(0.25 * sorted_x[:-2] + 0.5 * sorted_x[1:-1] + 0.25 * sorted_x[2:]),
        0.75 * sorted_x[-1] + 0.25 * sorted_x[-2],
    ]
    xr = np.linspace(0, 1, len(x) + 1)
    U = sorted(U)
    inds = np.searchsorted(xr, U, side="right") - 1
    interpolated = [desired_means[i] - (z_t[i] + z_t[i + 1]) / 2 for i in inds]
    y0, y1 = zip(
        *[
            (z_t[i] + interpolated[j], z_t[i + 1] + interpolated[j])
            for i, j in zip(inds, range(len(x)))
        ]
    )
    quantiles = [
        (y0[j] + ((U[j] - xr[i]) * (y1[j] - y0[j])) / (xr[i + 1] - xr[i]))
        for i, j in zip(inds, range(len(x)))
    ]
    replicate = [q for _, q in sorted(zip(indices, sorted(quantiles)))]
    return replicate
