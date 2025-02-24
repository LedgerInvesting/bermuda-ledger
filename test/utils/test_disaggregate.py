from datetime import date

import numpy as np

from bermuda import (
    CumulativeCell,
    Triangle,
    date_utils,
    disaggregate,
    disaggregate_experience,
)


def test_disaggregate():
    annual_tri = Triangle(
        [
            CumulativeCell(
                period_start=date(2020, 1, 1),
                period_end=date(2020, 12, 31),
                evaluation_date=date(2020, 12, 31),
                values={"paid_loss": 100},
            ),
            CumulativeCell(
                period_start=date(2020, 1, 1),
                period_end=date(2020, 12, 31),
                evaluation_date=date(2021, 12, 31),
                values={"paid_loss": 400},
            ),
            CumulativeCell(
                period_start=date(2021, 1, 1),
                period_end=date(2021, 12, 31),
                evaluation_date=date(2021, 12, 31),
                values={"paid_loss": 200},
            ),
        ]
    )
    quarterly = disaggregate(annual_tri)

    array_annual = annual_tri.derive_fields(
        paid_loss=lambda cel: np.array([cel["paid_loss"]] * 4)
    )
    array_quarterly = disaggregate(array_annual)

    assert isinstance(quarterly, Triangle)
    assert len(quarterly.periods) == 8
    assert len(quarterly.dev_lags()) == 8

    assert isinstance(array_quarterly, Triangle)
    assert len(array_quarterly.periods) == 8


def test_disaggregate_negative_lags():
    annual_tri = Triangle(
        [
            CumulativeCell(
                period_start=date(2020, 1, 1),
                period_end=date(2020, 12, 31),
                evaluation_date=date_utils.add_months(date(2020, 3, 31), extra_lag),
                values={"paid_loss": 100 + extra_paid},
            )
            for extra_lag, extra_paid in zip(range(0, 15, 3), range(0, 500, 100))
        ]
    )
    quarterly = disaggregate_experience(annual_tri, 3)

    lag_0_loss = sum(
        [
            cell["paid_loss"]
            for cell in annual_tri.filter(
                lambda cell: cell.evaluation_date == date(2020, 12, 31)
            )
        ]
    )

    assert quarterly.cells[0]["paid_loss"] == 100
    assert lag_0_loss == 400
