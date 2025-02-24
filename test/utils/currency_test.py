import datetime

import pytest

from bermuda import Cell, IncrementalCell, Metadata, Triangle
from bermuda.utils.currency import (
    DEFAULT_EXCHANGE_RATES,
    convert_currency,
    convert_to_dollars,
)

base_test_cell = Cell(
    period_start=datetime.date(2020, 1, 1),
    period_end=datetime.date(2020, 12, 31),
    evaluation_date=datetime.date(2020, 12, 31),
    values={
        "paid_loss": 100_000,
        "reported_loss": 200_000,
        "earned_premium": 300_000,
        "open_claims": 123,
    },
)

inc_test_cell = IncrementalCell(
    period_start=datetime.date(2020, 1, 1),
    period_end=datetime.date(2020, 12, 31),
    evaluation_date=datetime.date(2020, 12, 31),
    prev_evaluation_date=datetime.date(2019, 12, 31),
    values={
        "paid_loss": 100_000,
        "reported_loss": 200_000,
        "earned_premium": 300_000,
        "open_claims": 123,
    },
)

test_triangle = Triangle(
    [
        base_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.BRIT"}, currency="GBP")
        ),
        base_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.EURO"}, currency="EUR")
        ),
        base_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.USA"}, currency="USD")
        ),
    ]
)

test_inc_triangle = Triangle(
    [
        inc_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.BRIT"}, currency="GBP")
        ),
        inc_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.EURO"}, currency="EUR")
        ),
        inc_test_cell.replace(
            metadata=Metadata(details={"program_tag": "FOO.USA"}, currency="USD")
        ),
    ]
)


def test_currency_conversion():
    converted_tri = convert_to_dollars(test_triangle, {"EUR": 1.2, "GBP": 1.5})
    total_paid_loss = sum([cell["paid_loss"] for cell in converted_tri])
    total_earned_premium = sum([cell["earned_premium"] for cell in converted_tri])
    total_open_claims = sum([cell["open_claims"] for cell in converted_tri])

    converted_inc_tri = convert_to_dollars(test_inc_triangle, {"EUR": 1.2, "GBP": 1.5})
    total_paid_loss_inc = sum([cell["paid_loss"] for cell in converted_inc_tri])
    total_earned_premium_inc = sum(
        [cell["earned_premium"] for cell in converted_inc_tri]
    )
    total_open_claims_inc = sum([cell["open_claims"] for cell in converted_inc_tri])

    assert total_paid_loss == 370_000
    assert total_earned_premium == 370_000 * 3
    assert total_open_claims == 369

    assert total_paid_loss_inc == 370_000
    assert total_earned_premium_inc == 370_000 * 3
    assert total_open_claims_inc == 369

    converted_to_dollars_tri = convert_to_dollars(test_triangle)
    assert (
        converted_to_dollars_tri.cells[0].values["paid_loss"]
        == 100_000 * DEFAULT_EXCHANGE_RATES["GBP"]
    )
    assert (
        converted_to_dollars_tri.cells[1].values["paid_loss"]
        == 100_000 * DEFAULT_EXCHANGE_RATES["EUR"]
    )

    with pytest.raises(ValueError):
        convert_currency(test_triangle, "CAD", DEFAULT_EXCHANGE_RATES)

    test_tri_no_metadata = Triangle([base_test_cell])
    with pytest.raises(ValueError):
        convert_currency(test_tri_no_metadata, "EUR", DEFAULT_EXCHANGE_RATES)
