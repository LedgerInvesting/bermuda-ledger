from typing import Callable, Literal, Tuple, Union

import numpy as np

from bermuda.triangle import Triangle

__all__ = ["moment_match"]


def _sort_x_on_y_rank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # more efficient than y.argsort().argsort(), sorting only done once
    sort_y = y.argsort()
    rank_y = np.empty_like(sort_y)
    rank_y[sort_y] = np.arange(len(y))

    return np.array(sorted(x))[rank_y]


def _get_sample_moments(samples: np.ndarray) -> Tuple[float, float, int]:
    return (np.mean(samples), np.var(samples), len(samples))


def _sample_normal_dist(mu: float, sigma2: float, n: int) -> np.ndarray:
    return np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=n)


def _sample_lognormal_dist(mu: float, sigma2: float, n: int) -> np.ndarray:
    mean = (mu**2) / np.sqrt((mu**2) + sigma2)
    var = np.log(1 + sigma2 / (mu**2))
    return np.random.lognormal(mean=np.log(mean), sigma=np.sqrt(var), size=n)


def _sample_gamma_dist(mu: float, sigma2: float, n: int) -> np.ndarray:
    shape = mu**2 / sigma2
    rate = mu / sigma2
    return np.random.gamma(shape=shape, scale=1 / rate, size=n)


SAMPLE_DIST_dict = {
    "normal": _sample_normal_dist,
    "lognormal": _sample_lognormal_dist,
    "gamma": _sample_gamma_dist,
}


def _generate_samples(
    samples: Union[np.ndarray, float, int],
    sample_fn: Callable[[float, float, int], np.ndarray],
) -> Union[np.ndarray, float, int]:
    if type(samples) is not np.ndarray:
        return samples

    moments = _get_sample_moments(samples)

    return np.array(_sort_x_on_y_rank(sample_fn(*moments), samples))


def moment_match(
    triangle: Triangle,
    field_names: list[str],
    distribution: Literal["normal", "lognormal", "gamma"],
) -> Triangle:
    """Use method of moments matching to convert samples in a triangle from one distribution to another.

    Args:
        triangle: bermuda.triangle.Triangle with fields that need converted.
        field_names: List of field names to apply conversion to. Typical use cases include "reported_loss", "paid_loss",
        "incurred_loss", etc.
        distribution: Distribution to convert sample to. The empirical sample mean and variance are matched to moments of
        `distribution`, which is then sampled from.

    Raises:
        KeyError: if supplied `field_names` are not all contained in the input triangle.

    Returns:
        Triangle: bermuda.triangle.Triangle with `field_names` samples in each cell replaced with samples from `distribution`. If a
        field contains only scalar values in each cell, the scalar values are returned (no moment matching is done). Otherwise, if the
        field contains arrays of samples, moment matched samples will be returned (all with the same rank order as the original
        samples).
    """

    field_name_diff = list(set(field_names) - set(triangle.fields))
    if len(field_name_diff) > 0:
        raise KeyError(
            f"The following supplied field_names do not exist in `triangle`: {field_name_diff}"
        )

    for field in field_names:
        triangle = triangle.derive_fields(
            **{
                field: lambda ob: _generate_samples(
                    ob[field], SAMPLE_DIST_dict[distribution]
                )
            }
        )

    return triangle
