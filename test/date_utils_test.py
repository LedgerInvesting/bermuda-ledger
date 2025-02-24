from datetime import date, timedelta

import numpy as np
import numpy.testing as npt
import pytest

from bermuda.date_utils import (
    add_months,
    calculate_dev_lag,
    dev_lag_months,
    id_to_month,
    month_to_id,
    resolution_delta,
    standardize_resolution,
)


def test_dev_lag_months():
    npt.assert_almost_equal(dev_lag_months(date(2020, 1, 1), date(2020, 1, 1)), 0.0)
    npt.assert_almost_equal(dev_lag_months(date(2019, 12, 31), date(2020, 1, 31)), 1.0)
    npt.assert_approx_equal(dev_lag_months(date(2021, 1, 31), date(2021, 3, 31)), 2.0)
    npt.assert_approx_equal(
        dev_lag_months(date(2020, 1, 1), date(2020, 2, 1)), 1.00222469, significant=9
    )
    npt.assert_approx_equal(
        dev_lag_months(date(2020, 1, 1), date(2020, 1, 31)), 0.967741935, significant=9
    )


def test_add_integer_months():
    base = date(2020, 2, 29)
    assert add_months(base, 4.0) == date(2020, 6, 30)
    assert add_months(base, 10.0) == date(2020, 12, 31)
    assert add_months(base, 11.0) == date(2021, 1, 31)
    assert add_months(base, -5.0) == date(2019, 9, 30)


def test_add_float_months():
    base = date(2020, 2, 29)
    mid_month_base = date(1962, 5, 17)

    assert add_months(base, 9.999) == date(2020, 12, 31)
    assert add_months(base, 10.001) == date(2020, 12, 31)
    assert add_months(base, 10.02) == date(2021, 1, 1)
    assert add_months(base, 12.345) == date(2021, 3, 11)
    assert add_months(base, -0.456) == date(2020, 2, 16)

    assert add_months(mid_month_base, 3.0) == date(1962, 9, 16)
    assert add_months(mid_month_base, 7.1) == date(1963, 1, 20)
    assert add_months(mid_month_base, 6.4) == date(1962, 12, 29)


def test_add_inf_months():
    assert add_months(date(2000, 1, 1), np.inf) == date.max


def test_month_to_id():
    dates = [date(year, 1, 1) for year in range(1960, 1980)]
    ids = [month_to_id(day) for day in dates]
    dates_again = [id_to_month(idx) for idx in ids]

    assert dates == dates_again
    assert ids == [x * 12 for x in range(-10, 10)]


def test_calculate_dev_lag():
    assert calculate_dev_lag(date(2000, 12, 31), date.max, "timedelta") == timedelta.max
    assert calculate_dev_lag(date(2000, 12, 31), date.max, "day") == np.inf

    assert calculate_dev_lag(date(2000, 12, 31), date(2001, 1, 31), "day") == 31
    assert calculate_dev_lag(
        date(2000, 12, 31), date(2001, 1, 31), "timedelta"
    ) == timedelta(days=31)

    with pytest.raises(ValueError):
        calculate_dev_lag(date(2000, 12, 31), date(2001, 1, 31), "time")


def test_standardize_resolution():
    assert standardize_resolution((100, "day")) == (100, "day")
    assert standardize_resolution((3, "week")) == (21, "day")
    assert standardize_resolution((2, "quarter")) == (6, "month")

    with pytest.raises(ValueError):
        assert standardize_resolution((1, "period"))


def test_resolution_delta():
    assert resolution_delta(date(2000, 1, 1), (31, "day"), negative=False) == date(
        2000, 2, 1
    )
    assert resolution_delta(
        date(2000, 1, 1), (31, "timedelta"), negative=False
    ) == date(2000, 2, 1)
