from datetime import date

import pytest

from bermuda import CumulativeCell, Triangle, shift_origin


@pytest.fixture
def on_quarter_tri():
    return Triangle(
        cells=[
            CumulativeCell(
                period_start=date(2019, 1, 1),
                period_end=date(2019, 3, 31),
                evaluation_date=date(2019, 3, 31),
                values={},
            )
        ]
    )


@pytest.fixture
def off_quarter_tri():
    return Triangle(
        cells=[
            CumulativeCell(
                period_start=date(2019, 2, 1),
                period_end=date(2019, 4, 30),
                evaluation_date=date(2019, 4, 30),
                values={},
            )
        ]
    )


@pytest.fixture
def on_year_tri():
    return Triangle(
        cells=[
            CumulativeCell(
                period_start=date(2019, 1, 1),
                period_end=date(2019, 12, 31),
                evaluation_date=date(2019, 12, 31),
                values={},
            )
        ]
    )


@pytest.fixture
def off_year_tri():
    return Triangle(
        cells=[
            CumulativeCell(
                period_start=date(2019, 3, 1),
                period_end=date(2020, 2, 29),
                evaluation_date=date(2020, 2, 29),
                values={},
            )
        ]
    )


def test_shift_origin(on_quarter_tri, off_quarter_tri):
    shifted_tri = shift_origin(on_quarter_tri, off_quarter_tri)
    assert shifted_tri[0].period_start == date(2019, 2, 1)
    assert shifted_tri[0].period_end == date(2019, 4, 30)
    assert shifted_tri[0].evaluation_date == date(2019, 4, 30)


def test_shift_annual(on_year_tri, off_year_tri):
    shifted_tri = shift_origin(off_year_tri, on_year_tri)
    assert shifted_tri[0].period_start == date(2019, 1, 1)
    assert shifted_tri[0].period_end == date(2019, 12, 31)
    assert shifted_tri[0].evaluation_date == date(2019, 12, 31)


def test_round_trip(on_quarter_tri, off_quarter_tri):
    shifted_tri = shift_origin(on_quarter_tri, off_quarter_tri)
    shifted_back = shift_origin(shifted_tri, on_quarter_tri)
    assert shifted_back == on_quarter_tri


def test_idempotence(on_quarter_tri, off_quarter_tri):
    shifted_tri = shift_origin(on_quarter_tri, off_quarter_tri)
    shifted_again = shift_origin(shifted_tri, off_quarter_tri)
    assert shifted_again == off_quarter_tri


def test_mismatch_error(on_year_tri, off_quarter_tri):
    with pytest.raises(ValueError):
        shift_origin(on_year_tri, off_quarter_tri)
