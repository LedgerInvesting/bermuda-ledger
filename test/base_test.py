import datetime

import numpy as np
import pytest

from bermuda.base import Cell, IncrementalCell

unitary_ob = Cell(
    period_start=datetime.date(1970, 1, 1),
    period_end=datetime.date(1970, 3, 31),
    evaluation_date=datetime.date(1970, 3, 31),
    values={"paid_loss": 100},
)
joint_ob = Cell(
    period_start=datetime.date(1970, 1, 1),
    period_end=datetime.date(1970, 12, 31),
    evaluation_date=datetime.date(1971, 6, 30),
    values={"paid_loss": 100, "reported_loss": 200},
)

unitary_pred = Cell(
    period_start=datetime.date(1970, 2, 1),
    period_end=datetime.date(1970, 4, 30),
    evaluation_date=datetime.date(1970, 4, 30),
    values={"paid_loss": np.array([80, 85, 90, 95, 100])},
)
joint_pred = Cell(
    period_start=datetime.date(1970, 2, 1),
    period_end=datetime.date(1970, 4, 30),
    evaluation_date=datetime.date(1970, 4, 30),
    values={
        "paid_loss": np.array([80, 85, 90, 95, 100]),
        "reported_loss": np.array([195, 200, 205, 210, 215]),
    },
)

unitary_inc_ob = IncrementalCell(
    period_start=datetime.date(1970, 1, 1),
    period_end=datetime.date(1970, 3, 31),
    evaluation_date=datetime.date(1970, 6, 30),
    prev_evaluation_date=datetime.date(1970, 3, 31),
    values={"paid_loss": 100},
)
joint_inc_ob = IncrementalCell(
    period_start=datetime.date(1970, 1, 1),
    period_end=datetime.date(1970, 6, 30),
    evaluation_date=datetime.date(1970, 12, 31),
    prev_evaluation_date=datetime.date(1970, 6, 30),
    values={"paid_loss": 100, "reported_loss": 500},
)


def test_cell_properties():
    assert unitary_ob.period == (datetime.date(1970, 1, 1), datetime.date(1970, 3, 31))
    assert joint_ob.period == (datetime.date(1970, 1, 1), datetime.date(1970, 12, 31))
    assert unitary_inc_ob.period == (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 3, 31),
    )
    assert joint_inc_ob.period == (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 6, 30),
    )


def test_observation_conditions():
    # period_start must be > period_end
    with pytest.raises(ValueError):
        Cell(
            period_start=datetime.date(1970, 2, 1),
            period_end=datetime.date(1970, 1, 1),
            evaluation_date=datetime.date(1970, 1, 1),
            values={"paid_loss": 0},
        )

    # test evaluation_date before period_start
    with pytest.raises(ValueError):
        Cell(
            period_start=datetime.date(1980, 1, 1),
            period_end=datetime.date(1980, 2, 1),
            evaluation_date=datetime.date(1970, 1, 1),
            values={"paid_loss": 0},
        )

    # test evaluation_date finite
    with pytest.raises(ValueError):
        Cell(
            period_start=datetime.date(1980, 1, 1),
            period_end=datetime.date(1980, 2, 1),
            evaluation_date=datetime.date.max,
            values={"paid_loss": 0},
        )

    # test prev_evaluation_date after evaluation_date
    with pytest.raises(ValueError):
        IncrementalCell(
            period_start=datetime.date(1980, 1, 1),
            period_end=datetime.date(1980, 3, 31),
            evaluation_date=datetime.date(1980, 3, 31),
            prev_evaluation_date=datetime.date(1980, 6, 30),
            values={"paid_loss": 0},
        )


def test_observation_getitem():
    assert unitary_ob["paid_loss"] == 100
    assert joint_ob["paid_loss"] == 100
    assert joint_ob["reported_loss"] == 200
    assert unitary_inc_ob["paid_loss"] == 100
    assert joint_inc_ob["paid_loss"] == 100
    assert joint_inc_ob["reported_loss"] == 500

    with pytest.raises(KeyError):
        unitary_ob["reported_loss"]

    with pytest.raises(KeyError):
        joint_ob["askdfjalsef"]

    with pytest.raises(KeyError):
        unitary_inc_ob["reported_loss"]


def test_observation_derive_fields():
    derived_ob = joint_ob.derive_fields(
        case_reserve=lambda ob: ob["reported_loss"] - ob["paid_loss"],
        reserve_share=lambda ob: ob["case_reserve"] / ob["reported_loss"],
    )
    derived_inc_ob = joint_inc_ob.derive_fields(
        earned_premium=lambda ob: ob["reported_loss"] * 1.1
    )
    assert derived_ob["case_reserve"] == 100
    assert derived_ob["reserve_share"] == 0.5
    assert derived_inc_ob["earned_premium"] == 550


def test_observation_coordinates():
    true_unitary_ob_coords = (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 3, 31),
        datetime.date(1970, 3, 31),
    )
    assert unitary_ob.coordinates == true_unitary_ob_coords

    true_joint_ob_coords = (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 12, 31),
        datetime.date(1971, 6, 30),
    )
    assert joint_ob.coordinates == true_joint_ob_coords

    true_unitary_inc_ob_coords = (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 3, 31),
        datetime.date(1970, 6, 30),
        datetime.date(1970, 3, 31),
    )
    assert unitary_inc_ob.coordinates == true_unitary_inc_ob_coords

    true_joint_inc_ob_coord = (
        datetime.date(1970, 1, 1),
        datetime.date(1970, 6, 30),
        datetime.date(1970, 12, 31),
        datetime.date(1970, 6, 30),
    )
    assert joint_inc_ob.coordinates == true_joint_inc_ob_coord


def test_print_nofail():
    joint_ob.__str__()
    joint_pred.__str__()
    joint_inc_ob.__str__()
