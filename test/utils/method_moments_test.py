import datetime

import numpy as np
from scipy.stats import gamma, lognorm, norm, rankdata

from bermuda import Cell, Triangle, moment_match
from bermuda.utils.method_moments import (
    _get_sample_moments,
    _sample_gamma_dist,
    _sample_lognormal_dist,
    _sample_normal_dist,
    _sort_x_on_y_rank,
)

DETAILS = {"line_of_business": "B1", "risk_basis": "Accident"}

SAMPLE_MEAN = 1
SAMPLE_SIGMA = 1
SAMPLES = norm.rvs(loc=SAMPLE_MEAN, scale=SAMPLE_SIGMA, size=100_000)

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

TEST_TRI = Triangle(
    [
        Cell(
            period_start=datetime.date(2020, 1, 1),
            period_end=datetime.date(2020, 3, 31),
            evaluation_date=datetime.date(2020, 3, 31),
            values={
                "reported_loss": np.array([210, 200, 250, 220]),
                "paid_loss": 1,
                "random_field": 0,
            },
        ),
        Cell(
            period_start=datetime.date(2020, 1, 1),
            period_end=datetime.date(2020, 3, 31),
            evaluation_date=datetime.date(2020, 6, 30),
            values={
                "reported_loss": np.array([270, 250, 260, 280]),
                "paid_loss": 2,
                "random_field": 0,
            },
        ),
        Cell(
            period_start=datetime.date(2020, 1, 1),
            period_end=datetime.date(2020, 3, 31),
            evaluation_date=datetime.date(2020, 9, 30),
            values={
                "reported_loss": np.array([310, 320, 305, 290]),
                "paid_loss": 3,
                "random_field": 0,
            },
        ),
        Cell(
            period_start=datetime.date(2020, 4, 1),
            period_end=datetime.date(2020, 6, 30),
            evaluation_date=datetime.date(2020, 6, 30),
            values={
                "reported_loss": np.array([400, 390, 410, 420]),
                "paid_loss": 4,
                "random_field": 0,
            },
        ),
        Cell(
            period_start=datetime.date(2020, 4, 1),
            period_end=datetime.date(2020, 6, 30),
            evaluation_date=datetime.date(2020, 9, 30),
            values={
                "reported_loss": np.array([480, 510, 500, 490]),
                "paid_loss": 5,
                "random_field": 0,
            },
        ),
        Cell(
            period_start=datetime.date(2020, 7, 1),
            period_end=datetime.date(2020, 9, 30),
            evaluation_date=datetime.date(2020, 9, 30),
            values={
                "reported_loss": np.array([500, 505, 490, 520]),
                "paid_loss": 6,
                "random_field": 0,
            },
        ),
    ]
)


def test_sort_x_on_y_rank() -> None:
    x1 = np.array([-1.03, 1.16, 0.58, -0.04, -0.19])
    y1 = np.array([-0.44, -1.28, 0.91, 1.08, 0.16])

    x2 = np.array([0.41, -1.13, -0.39, -0.57, -0.75])
    y2 = np.array([-2.25, -0.58, -0.19, -0.93, -1.37])

    expected1 = np.array([-0.19, -1.03, 0.58, 1.16, -0.04])
    expected2 = np.array([-1.13, -0.39, 0.41, -0.57, -0.75])

    np.testing.assert_array_equal(_sort_x_on_y_rank(x1, y1), expected1)
    np.testing.assert_array_equal(_sort_x_on_y_rank(x2, y2), expected2)


def test_sample_normal_dist() -> None:
    moments = _get_sample_moments(SAMPLES)
    new_samples = _sample_normal_dist(*moments)

    loc = SAMPLE_MEAN
    scale = SAMPLE_SIGMA

    quantiles = np.quantile(norm.cdf(loc=loc, scale=scale, x=new_samples), q=QUANTILES)

    # if new samples follow the cdf of implied by method of moments, empirical
    # quantiles we get back should match those we pass in (quantile function is inverse of cdf)
    np.testing.assert_allclose(quantiles, QUANTILES, atol=0.01)


def test_sample_lognormal_dist() -> None:
    moments = _get_sample_moments(SAMPLES)
    new_samples = _sample_lognormal_dist(*moments)

    mean = (SAMPLE_MEAN**2) / np.sqrt((SAMPLE_MEAN**2) + SAMPLE_SIGMA**2)
    var = np.log(1 + SAMPLE_SIGMA**2 / (SAMPLE_MEAN**2))

    quantiles = np.quantile(
        lognorm.cdf(scale=mean, s=np.sqrt(var), x=new_samples), q=QUANTILES
    )

    # if new samples follow the cdf of implied by method of moments, empirical
    # quantiles we get back should match those we pass in (quantile function is inverse of cdf)
    np.testing.assert_allclose(quantiles, QUANTILES, atol=0.01)


def test_sample_gamma_dist() -> None:
    moments = _get_sample_moments(SAMPLES)
    new_samples = _sample_gamma_dist(*moments)

    shape = SAMPLE_MEAN**2 / SAMPLE_SIGMA**2
    rate = SAMPLE_MEAN / SAMPLE_SIGMA**2

    quantiles = np.quantile(
        gamma.cdf(a=shape, scale=1 / rate, x=new_samples), q=QUANTILES
    )

    # if new samples follow the cdf of implied by method of moments, empirical
    # quantiles we get back should match those we pass in (quantile function is inverse of cdf)
    np.testing.assert_allclose(quantiles, QUANTILES, atol=0.01)


def test_moment_match() -> None:
    new_tri = moment_match(
        TEST_TRI, field_names=["reported_loss"], distribution="normal"
    )

    old_vals = [ob["reported_loss"] for ob in TEST_TRI]
    new_vals = [ob["reported_loss"] for ob in new_tri]

    # should return triangle
    assert isinstance(new_tri, Triangle)

    # new sampled values should be differnet from previous values
    assert not any([all(x == y) for x, y in zip(new_vals, old_vals)])

    # rank order of new and old values should match
    assert all([all(rankdata(x) == rankdata(y)) for x, y in zip(new_vals, old_vals)])


def test_moment_match_scalar_field() -> None:
    new_tri = moment_match(TEST_TRI, field_names=["paid_loss"], distribution="normal")

    old_vals = [ob["paid_loss"] for ob in TEST_TRI]
    new_vals = [ob["paid_loss"] for ob in new_tri]

    # if field_names is a scalar, we should get back the same values
    assert all([x == y for x, y in zip(new_vals, old_vals)])


def test_moment_match_multiple_fields() -> None:
    field_names = ["paid_loss", "reported_loss"]
    new_tri = moment_match(TEST_TRI, field_names=field_names, distribution="normal")

    # really just ensuring that this function runs without error.
    # field names should be equal before and after
    assert set(field_names).issubset(set(new_tri.fields))


def test_moment_match_keep_other_fields_as_is() -> None:
    field_names = ["paid_loss", "reported_loss"]
    new_tri = moment_match(TEST_TRI, field_names=field_names, distribution="normal")

    # should keep random_field to preserve all information
    assert "random_field" in new_tri.fields
