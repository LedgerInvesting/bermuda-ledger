import datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from bermuda import Cell, Metadata, Triangle, paid_bs_adjustment, reported_bs_adjustment
from bermuda.date_utils import dev_lag_months

N_SLICES = 3


def _get_multi_slice_tri(tri: Triangle, n_slices: int = N_SLICES):
    return sum(
        [
            tri.replace(metadata=Metadata(details={"program_tag": f"TEST.N{idx}"}))
            for idx in range(n_slices)
        ]
    )


@pytest.fixture
def triangle_cumulative():
    tri = Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2020, 12, 31),
                values={
                    "cwp_claims": 73,
                    "open_claims": 25,
                    "paid_loss": 25_000,
                    "reported_loss": 50_000,
                },
            ),
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2021, 12, 31),
                values={
                    "cwp_claims": 85,
                    "open_claims": 15,
                    "paid_loss": 120_000,
                    "reported_loss": 165_000,
                },
            ),
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2022, 12, 31),
                values={
                    "cwp_claims": 92,
                    "open_claims": 5,
                    "paid_loss": 190_000,
                    "reported_loss": 215_000,
                },
            ),
            Cell(
                period_start=datetime.date(2021, 1, 1),
                period_end=datetime.date(2021, 12, 31),
                evaluation_date=datetime.date(2021, 12, 31),
                values={
                    "cwp_claims": 80,
                    "open_claims": 25,
                    "paid_loss": 27_500,
                    "reported_loss": 64_625,
                },
            ),
            Cell(
                period_start=datetime.date(2021, 1, 1),
                period_end=datetime.date(2021, 12, 31),
                evaluation_date=datetime.date(2022, 12, 31),
                values={
                    "cwp_claims": 100,
                    "open_claims": 15,
                    "paid_loss": 132_000,
                    "reported_loss": 181_500,
                },
            ),
            Cell(
                period_start=datetime.date(2022, 1, 1),
                period_end=datetime.date(2022, 12, 31),
                evaluation_date=datetime.date(2022, 12, 31),
                values={
                    "cwp_claims": 95,
                    "open_claims": 25,
                    "paid_loss": 30_250,
                    "reported_loss": 64_625,
                },
            ),
        ]
    )

    return _get_multi_slice_tri(tri, N_SLICES)


@pytest.fixture
def triangle_ult_claim_count():
    tri = Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2022, 12, 31),
                values={
                    "reported_claims": 100,
                },
            ),
            Cell(
                period_start=datetime.date(2021, 1, 1),
                period_end=datetime.date(2021, 12, 31),
                evaluation_date=datetime.date(2021, 12, 31),
                values={
                    "reported_claims": 120,
                },
            ),
            Cell(
                period_start=datetime.date(2022, 1, 1),
                period_end=datetime.date(2022, 12, 31),
                evaluation_date=datetime.date(2022, 12, 31),
                values={
                    "reported_claims": np.array([140, 130, 150]),
                },
            ),
        ]
    )

    return _get_multi_slice_tri(tri, N_SLICES)


def test_paid_berquist_sherman(triangle_cumulative, triangle_ult_claim_count):
    tri_adjusted = paid_bs_adjustment(triangle_cumulative, triangle_ult_claim_count)

    expected_paid_loss = [
        (0 * (73 / 100 - 95 / 140) + 25_000 * (95 / 140 - 0)) / (73 / 100 - 0),
        (25_000 * (85 / 100 - 100 / 120) + 120_000 * (100 / 120 - 95 / 140))
        / (85 / 100 - 95 / 140),
        190_000,
        (27_500 * (100 / 120 - 95 / 140) + 132_000 * (95 / 140 - 80 / 120))
        / (100 / 120 - 80 / 120),
        132_000,
        30_250,
    ]

    assert_array_equal(
        expected_paid_loss * N_SLICES,
        [ob["paid_loss"] for ob in tri_adjusted],
    )


def test_paid_berquist_sherman_ragged_right_edge(
    triangle_cumulative,
    triangle_ult_claim_count,
):
    # Logic should still work when the latest period has more than one dev lag e.g.:
    #
    #   CY1 * * * * *
    #   CY2 * * * *
    #   CY3 * * *
    #   CY4 * *
    #
    # In `right_edge` doesn't get a selected disposal rate for all dev lags in the
    # triangle (i.e lag 0 disposal rate will be unknown).
    tri_adjusted = paid_bs_adjustment(
        triangle_cumulative.clip(max_period=datetime.date(2021, 12, 31)),
        triangle_ult_claim_count,
    )

    expected_paid_loss = [
        0 * (0.73 - 2 / 3) / (0.73 - 0)
        + triangle_cumulative[0]["paid_loss"] * (2 / 3 - 0) / (0.73 - 0),
        triangle_cumulative[0]["paid_loss"] * (0.85 - 5 / 6) / (0.85 - 2 / 3)
        + triangle_cumulative[1]["paid_loss"] * (5 / 6 - 2 / 3) / (0.85 - 2 / 3),
        190_000,
        27_500,
        132_000,
    ]

    assert_allclose(
        expected_paid_loss * N_SLICES,
        [ob["paid_loss"] for ob in tri_adjusted],
    )


def test_reported_berquist_sherman(triangle_cumulative):
    tri_adjusted = reported_bs_adjustment(triangle_cumulative)

    expected_average_case_os = [
        (64_625 - 30_250) / 25 / 1.0**2,
        (181_500 - 132_000) / 15 / 1.0,
        (215_000 - 190_000) / 5,
        (64_625 - 30_250) / 25 / 1.0,
        (181_500 - 132_000) / 15,
        (64_625 - 30_250) / 25,
    ]

    assert_array_equal(
        expected_average_case_os * N_SLICES,
        [ob["average_case_os"] for ob in tri_adjusted],
    )

    expected_reported_loss = [
        expected_average_case_os[0] * 25 + 25_000,
        expected_average_case_os[1] * 15 + 120_000,
        expected_average_case_os[2] * 5 + 190_000,
        expected_average_case_os[3] * 25 + 27_500,
        expected_average_case_os[4] * 15 + 132_000,
        expected_average_case_os[5] * 25 + 30_250,
    ]

    assert_array_equal(
        expected_reported_loss * N_SLICES,
        [ob["reported_loss"] for ob in tri_adjusted],
    )


def test_reported_bs_w_sev_trend_calculation_all(triangle_cumulative):
    expected_sev_trend_factors = [
        ((27_500 / 80) / (25_000 / 73))
        ** (
            dev_lag_months(
                triangle_cumulative[0].period_end, triangle_cumulative[3].period_end
            )
            / 12
        ),
        (132_000 / 100)
        / (120_000 / 85)
        ** (
            dev_lag_months(
                triangle_cumulative[1].period_end, triangle_cumulative[4].period_end
            )
            / 12
        ),
        (30_250 / 95)
        / (27_500 / 80)
        ** (
            dev_lag_months(
                triangle_cumulative[3].period_end, triangle_cumulative[5].period_end
            )
            / 12
        ),
    ] * N_SLICES
    mean_sev_trend_factor = sum(expected_sev_trend_factors) / len(
        expected_sev_trend_factors
    )
    expected_average_case_os = [
        (64_625 - 30_250) / 25 / mean_sev_trend_factor**2,
        (181_500 - 132_000) / 15 / mean_sev_trend_factor,
        (215_000 - 190_000) / 5,
        (64_625 - 30_250) / 25 / mean_sev_trend_factor,
        (181_500 - 132_000) / 15,
        (64_625 - 30_250) / 25,
    ] * N_SLICES

    tri_adjusted = reported_bs_adjustment(triangle_cumulative, sev_trend_method="all")

    assert_array_almost_equal(
        expected_average_case_os,
        [ob["average_case_os"] for ob in tri_adjusted],
    )
    assert_array_almost_equal(
        [
            case * ob["open_claims"] + ob["paid_loss"]
            for case, ob in zip(expected_average_case_os, triangle_cumulative)
        ],
        [ob["reported_loss"] for ob in tri_adjusted],
    )


def test_reported_bs_w_sev_trend_calculation_latest(triangle_cumulative):
    expected_sev_trend_factors = [
        (132_000 / 100)
        / (120_000 / 85)
        ** (
            dev_lag_months(
                triangle_cumulative[1].period_end, triangle_cumulative[4].period_end
            )
            / 12
        ),
        (30_250 / 95)
        / (27_500 / 80)
        ** (
            dev_lag_months(
                triangle_cumulative[3].period_end, triangle_cumulative[5].period_end
            )
            / 12
        ),
    ]

    tri_adjusted = reported_bs_adjustment(
        triangle_cumulative, sev_trend_method="latest"
    )
    mean_sev_trend = sum(expected_sev_trend_factors) / 2

    expected_average_case_os = [
        (64_625 - 30_250) / 25 / mean_sev_trend**2,
        (181_500 - 132_000) / 15 / mean_sev_trend,
        (215_000 - 190_000) / 5,
        (64_625 - 30_250) / 25 / mean_sev_trend,
        (181_500 - 132_000) / 15,
        (64_625 - 30_250) / 25,
    ]

    assert_array_almost_equal(
        [x for x in expected_average_case_os] * N_SLICES,
        [ob["average_case_os"] for ob in tri_adjusted],
    )
    assert_array_almost_equal(
        [
            case * ob["open_claims"] + ob["paid_loss"]
            for case, ob in zip(
                expected_average_case_os * N_SLICES, triangle_cumulative
            )
        ],
        [ob["reported_loss"] for ob in tri_adjusted],
    )
