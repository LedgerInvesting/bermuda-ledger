import datetime
import re

import numpy as np
import pytest

from bermuda import Cell
from bermuda.base.cell import format_value
from bermuda.base.incremental import IncrementalCell

from .testing_utils import is_valid_html

test_cell = Cell(
    period_start=datetime.date(2017, 1, 1),
    period_end=datetime.date(2017, 12, 31),
    evaluation_date=datetime.date(2018, 12, 31),
    values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
)
test_inc_cell = IncrementalCell(
    period_start=datetime.date(2017, 1, 1),
    period_end=datetime.date(2017, 12, 31),
    evaluation_date=datetime.date(2018, 12, 31),
    prev_evaluation_date=datetime.date(2017, 12, 31),
    values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
)


def test_cell_value_type_constraints():
    # We shouldn't be able to create a stringly-typed Cell value
    with pytest.raises(Exception):
        Cell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            evaluation_date=datetime.date(2019, 12, 31),
            values={"terrible": "error!"},
        )

    # We shouldn't be able to break typing restrictions via replace
    with pytest.raises(Exception):
        test_cell.replace(values={"paid_loss": "oh no!"})
    # We also shouldn't be able to break typing restrictions via derive_fields
    with pytest.raises(Exception):
        test_cell.derive_fields(reported_loss=lambda cell: "break")

    # Value types shouldn't contain non-float values
    with pytest.raises(ValueError):
        Cell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            evaluation_date=datetime.date(2018, 12, 31),
            values={"paid_loss": np.array(["one", "two", "three"])},
        )


def test_cell_date_type_constraint():
    # Shouldn't be able to create a Cell with stringly-typed dates
    with pytest.raises(TypeError):
        Cell(
            period_start="2017-01-01",
            period_end="2017-12-31",
            evaluation_date="2018-12-31",
            values={
                "paid_loss": 10_000,
                "reported_loss": 20_000,
                "earned_premium": 50e3,
            },
        )

    # If we use datetime instead of date, it should still work but be converted to date
    test_cell = Cell(
        period_start=datetime.datetime(2017, 1, 1),
        period_end=datetime.datetime(2017, 12, 31),
        evaluation_date=datetime.datetime(2018, 12, 31),
        values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
    )

    assert test_cell.period_start == datetime.date(2017, 1, 1)
    assert test_cell.period_end == datetime.date(2017, 12, 31)
    assert test_cell.evaluation_date == datetime.date(2018, 12, 31)

    # Additionally, we shouldn't be able to replace these values with a string or datetime
    with pytest.raises(Exception):
        test_cell.replace(period_start="2017-01-01")
    with pytest.raises(Exception):
        test_cell.replace(period_end="2017-12-31")
    with pytest.raises(Exception):
        test_cell.replace(evaluation_date="2018-12-31")

    # Replacing them with a datetime should result in expected behavior
    test_cell.replace(period_start=datetime.datetime(2017, 1, 1))
    test_cell.replace(period_end=datetime.datetime(2017, 12, 31))
    test_cell.replace(evaluation_date=datetime.datetime(2018, 12, 31))
    assert test_cell.period_start == datetime.date(2017, 1, 1)
    assert test_cell.period_end == datetime.date(2017, 12, 31)
    assert test_cell.evaluation_date == datetime.date(2018, 12, 31)


def test_cell_invalid_date_args():
    # test when period_end < period_start
    with pytest.raises(ValueError, match=r"`period_end` .* >= `period_start`"):
        Cell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2016, 12, 31),
            evaluation_date=datetime.date(2018, 12, 31),
            values={
                "paid_loss": 10_000,
                "reported_loss": 20_000,
                "earned_premium": 50e3,
            },
        )

    # test when evaluation_date < period_start
    with pytest.raises(ValueError, match=r"`evaluation_date` .* >= `period_start`"):
        Cell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            evaluation_date=datetime.date(2016, 12, 31),
            values={
                "paid_loss": 10_000,
                "reported_loss": 20_000,
                "earned_premium": 50e3,
            },
        )

    # test with max evaluation_date
    with pytest.raises(ValueError, match="datetime.date.max"):
        Cell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            evaluation_date=datetime.date.max,
            values={
                "paid_loss": 10_000,
                "reported_loss": 20_000,
                "earned_premium": 50e3,
            },
        )


def test_select():
    small_cell = test_cell.select(["paid_loss", "earned_premium"])
    small_inc_cell = test_inc_cell.select(["paid_loss", "earned_premium"])
    assert sorted(small_cell.values.keys()) == ["earned_premium", "paid_loss"]
    assert sorted(small_inc_cell.values.keys()) == ["earned_premium", "paid_loss"]


def test_add_statics():
    static_cell = test_cell.derive_fields(
        earned_premium=lambda ob: 80_000,
        written_premium=lambda ob: 90_000,
    )
    rich_cell = test_cell.add_statics(
        static_cell, ["earned_premium", "written_premium"]
    )
    static_inc_cell = test_inc_cell.derive_fields(
        earned_premium=lambda ob: 80_000,
        written_premium=lambda ob: 90_000,
    )
    rich_inc_cell = test_inc_cell.add_statics(
        static_inc_cell, ["earned_premium", "written_premium"]
    )

    assert rich_cell["earned_premium"] == 80_000
    assert rich_cell["written_premium"] == 90_000
    assert rich_inc_cell["earned_premium"] == 80_000
    assert rich_inc_cell["written_premium"] == 90_000


def test_equality():
    alt_cell_1 = test_cell.derive_fields(paid_loss=lambda ob: ob["paid_loss"])
    alt_cell_2 = test_cell.derive_fields(paid_loss=lambda ob: 40_000)
    alt_cell_3 = test_cell.derive_fields(written_premium=lambda ob: 90_000)
    alt_inc_cell_1 = test_inc_cell.derive_fields(paid_loss=lambda ob: ob["paid_loss"])
    alt_inc_cell_2 = test_inc_cell.derive_fields(paid_loss=lambda ob: 40_000)
    alt_inc_cell_3 = test_inc_cell.derive_fields(written_premium=lambda ob: 90_000)

    assert test_cell == alt_cell_1
    assert test_cell != alt_cell_2
    assert test_cell != alt_cell_3
    assert test_inc_cell == alt_inc_cell_1
    assert test_inc_cell != alt_inc_cell_2
    assert test_inc_cell != alt_inc_cell_3


def test_incremental_eval_lag():
    assert test_inc_cell.eval_lag(unit="month") == 12.0


def test_incremental_init_eval_dates():
    # Should raise error when prev_evaluation_date == evaluation_date
    with pytest.raises(ValueError):
        IncrementalCell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            prev_evaluation_date=datetime.date(2019, 12, 31),
            evaluation_date=datetime.date(2019, 12, 31),
            values={"paid_loss": 10_000},
        )

    # Should raise error when prev_evaluation_date > evaluation_date
    with pytest.raises(ValueError):
        IncrementalCell(
            period_start=datetime.date(2017, 1, 1),
            period_end=datetime.date(2017, 12, 31),
            prev_evaluation_date=datetime.date(2020, 12, 31),
            evaluation_date=datetime.date(2019, 12, 31),
            values={"paid_loss": 10_000},
        )


def test_cell_repr_html():
    # It's brittle to make assertions on the HTML structure, so for now we can
    # Just check that the return value is valid HTML.
    assert is_valid_html(test_cell._repr_html_()), (
        "string returned from _repr_html_ is not valid html"
    )

    # test _repr_html_ with ndarray values
    test_inc_cell_np = Cell(
        period_start=datetime.date(2017, 1, 1),
        period_end=datetime.date(2017, 12, 31),
        evaluation_date=datetime.date(2018, 12, 31),
        values={
            "paid_loss": np.array([10_000]),
            "reported_loss": np.array([20_000]),
            "earned_premium": np.array([50e3]),
        },
    )

    assert is_valid_html(test_inc_cell_np._repr_html_()), (
        "string returned from _repr_html_ is not valid html"
    )


def test_incremental_cell_repr_html():
    # It's brittle to make assertions on the HTML structure, so for now we can
    # Iust check that the return value is valid HTML.
    assert is_valid_html(test_inc_cell._repr_html_()), (
        "string returned from _repr_html_ is not valid html"
    )

    # test _repr_html_ with ndarray values
    test_inc_cell_np = IncrementalCell(
        period_start=datetime.date(2017, 1, 1),
        period_end=datetime.date(2017, 12, 31),
        evaluation_date=datetime.date(2018, 12, 31),
        prev_evaluation_date=datetime.date(2017, 12, 31),
        values={
            "paid_loss": np.array([10_000]),
            "reported_loss": np.array([20_000]),
            "earned_premium": np.array([50e3]),
        },
    )

    assert is_valid_html(test_inc_cell_np._repr_html_()), (
        "string returned from _repr_html_ is not valid html"
    )


def test_format_value():
    assert format_value(123.0) == "123.0"
    assert format_value(2345.0) == "2.345K"
    assert format_value(34.56e6) == "34.56M"
    assert format_value(-1e7) == "-10.0M"
    assert format_value(5.6e9) == "5.6B"
    assert format_value(6.78923e12, 3) == "6.79T"
    assert format_value(3.45e34) == "34.5e33"
    assert format_value(0.23456) == "0.2346"
    assert format_value(123) == "123"

    with pytest.raises(Exception):
        format_value(123, 2)


def test_cell_period():
    assert test_cell.period == (datetime.date(2017, 1, 1), datetime.date(2017, 12, 31))


def test_cell_coordinates():
    # NOTE: this is just to get code coverage for the `coordinates` property.
    #       This sort of idiom is unlikely to be useful in practical code.
    assert test_cell.coordinates == (
        datetime.date(2017, 1, 1),
        datetime.date(2017, 12, 31),
        datetime.date(2018, 12, 31),
    )


def test_cell_get_item():
    # test with valid key
    assert test_cell["paid_loss"] == 10000

    # test with unrecognized key
    with pytest.raises(KeyError, match=r"Field .* does not exist"):
        test_cell["unknown_key"]


def test_cell_contains():
    # test with valid key
    assert "paid_loss" in test_cell
    # test with unknown key
    assert "unknown_key" not in test_cell


def test_cell_lt_comparison():
    cell_1 = Cell(
        period_start=datetime.date(2016, 1, 1),
        period_end=datetime.date(2016, 12, 31),
        evaluation_date=datetime.date(2018, 12, 31),
        values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
    )

    cell_2 = Cell(
        period_start=datetime.date(2017, 1, 1),
        period_end=datetime.date(2017, 12, 31),
        evaluation_date=datetime.date(2018, 12, 31),
        values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
    )

    assert cell_1 < cell_2

    # NOTE: period_start is before cell_2, but evaluation_date and
    #       period_end are after cell_2.
    cell_3 = Cell(
        period_start=datetime.date(2016, 1, 1),
        period_end=datetime.date(2018, 12, 31),
        evaluation_date=datetime.date(2019, 12, 31),
        values={"paid_loss": 10_000, "reported_loss": 20_000, "earned_premium": 50e3},
    )

    assert cell_3 < cell_2


def test_cell_repr():
    # FIXME: I'm open to better ways to test this __repr__ method
    res = repr(test_cell)
    assert re.match(r"Cell(.*)", res)


def test_cell_derive_metadata():
    derived_cell = test_cell.derive_metadata(
        country="CA",
        some_key="something else",
        some_other_key=lambda c: "a string",
    )

    # we expect all keys to be the same except for "country" (modified above)
    # and "details" (since additional fields are added to details)
    assert all(
        [
            getattr(test_cell.metadata, k) == getattr(derived_cell.metadata, k)
            for k in test_cell.metadata.as_dict().keys()
            if k not in ["country", "details"]
        ]
    )

    assert derived_cell.country == "CA"
    assert derived_cell.details["some_key"] == "something else"
    assert derived_cell.details["some_other_key"] == "a string"


def test_sketchy_replacement():
    start_cell = Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 12, 31),
        evaluation_date=datetime.date(2020, 12, 31),
        values={},
    )

    new_cell = start_cell.replace(
        period_start=datetime.date(2021, 1, 1),
        period_end=datetime.date(2021, 12, 31),
        evaluation_date=datetime.date(2021, 12, 31),
    )
    assert new_cell.period_start == datetime.date(2021, 1, 1)
    assert new_cell.period_end == datetime.date(2021, 12, 31)


def test_cell_type_check_int64():
    # will fail if int64 or float64 do not pass type checks per base cell class
    cell_int64 = test_cell.derive_fields(
        reported_loss=lambda ob: np.int64(ob["reported_loss"])
    )
    cell_float64 = test_cell.derive_fields(
        reported_loss=lambda ob: np.float64(ob["reported_loss"])
    )

    assert isinstance(cell_int64, Cell)
    assert isinstance(cell_float64, Cell)


def test_cell_type_check_None():
    # will fail if None does not pass type checks per base cell class
    cell_None = test_cell.derive_fields(reported_loss=lambda ob: None)

    assert isinstance(cell_None, Cell)
