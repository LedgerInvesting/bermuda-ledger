import numpy as np

from ..triangle import Cell, Triangle


def thin(triangle: Triangle, num_samples: int, seed: int = None) -> Triangle:
    """'Thin out' a triangle by replacing all array-typed values with a
    random subset of values. For example, given a triangle `pred_tri` with
    10,000 samples per value, `thin(pred_tri, 500)` will return
    a triangle with 500 samples per value.

    The thinning process takes the same random subset of samples from each
    value and preserves the relative order of the subsamples, so that
    correlations between values (both within and across cells) are preserved.

    Arguments:
        triangle: The triangle to thin.
        num_samples: The number of samples of each array-valued value to return.
        seed: The seed of the random number generator.
    Returns:
        A copy of the triangle with `num_samples` random samples.
    """

    if triangle.num_samples < num_samples:
        raise ValueError(
            f"Cannot return {num_samples} samples, "
            f"the source triangle only has {num_samples} samples"
        )
    elif triangle.num_samples == num_samples:
        return triangle
    else:
        rng = np.random.default_rng(seed)
        ndxs = rng.choice(triangle.num_samples, num_samples, False)
        return Triangle([_thin_cell(cell, ndxs) for cell in triangle])


def _thin_cell(cell: Cell, ndxs: np.ndarray) -> Cell:
    new_values = {
        k: v[ndxs] if isinstance(v, np.ndarray) and len(v) > 1 else v
        for k, v in cell.values.items()
    }
    return cell.replace(values=new_values)
