import datetime

import numpy as np
import pytest

from bermuda import Cell, Triangle, weight_geometric_decay

TOLERANCE = 1e-4  # Tolerance for checking correct values - relative error
base_cells = [
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 3, 31),
        evaluation_date=datetime.date(2019, 3, 31),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 3, 31),
        evaluation_date=datetime.date(2019, 6, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 3, 31),
        evaluation_date=datetime.date(2019, 9, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 4, 1),
        period_end=datetime.date(2019, 6, 30),
        evaluation_date=datetime.date(2019, 6, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 4, 1),
        period_end=datetime.date(2019, 6, 30),
        evaluation_date=datetime.date(2019, 9, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 7, 1),
        period_end=datetime.date(2019, 9, 30),
        evaluation_date=datetime.date(2019, 9, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
    Cell(
        period_start=datetime.date(2019, 7, 1),
        period_end=datetime.date(2019, 9, 30),
        evaluation_date=datetime.date(2019, 10, 30),
        values={
            "paid_loss": np.array(1),
            "reported_loss": np.array(10),
            "earned_premium": 1000,
        },
    ),
]

test_triangle = Triangle(base_cells[0:6])
non_regular_triangle = Triangle([base_cells[0], base_cells[2], base_cells[6]])


def test_geom_decay_weight():
    weight_as_field_arg = False

    # Regular, standard quarterly triangle case
    test_triangle_weighted = weight_geometric_decay(
        test_triangle, 0.9, basis="experience", weight_as_field=weight_as_field_arg
    )
    paid_losses = [cel["paid_loss"] for cel in test_triangle_weighted]
    reported_losses = [cel["reported_loss"] for cel in test_triangle_weighted]
    earned_prem = [cel["earned_premium"] for cel in test_triangle_weighted]

    # regular, standard quarterly triangle with evaluation date weighting
    test_triangle_evaluation = weight_geometric_decay(
        test_triangle, 0.9, basis="evaluation", weight_as_field=True
    )

    derived_weights_evaluation = [
        cel["geometric_weight"] for cel in test_triangle_evaluation
    ]
    correct_weights_evaluation = [
        0.9 ** (2 / 4),
        0.9 ** (1 / 4),
        1,
        0.9 ** (1 / 4),
        1,
        1,
    ]
    # If loss fields are specified
    test_triangle_lf = weight_geometric_decay(
        test_triangle,
        0.9,
        tri_fields="paid_loss",
        basis="experience",
        weight_as_field=weight_as_field_arg,
    )
    paid_losses_lf = [cel["paid_loss"] for cel in test_triangle_lf]
    reported_losses_lf = [cel["reported_loss"] for cel in test_triangle_lf]

    # Non-regular triangle
    test_nr_triangle = weight_geometric_decay(
        non_regular_triangle,
        0.9,
        basis="experience",
        weight_as_field=weight_as_field_arg,
    )
    paid_losses_nr = [cel["paid_loss"] for cel in test_nr_triangle]

    correct_paids = np.array([0.9 ** (2 / 4)] * 3 + [0.9 ** (1 / 4)] * 2 + [1])
    correct_reported = correct_paids * 10
    correct_ep = correct_paids * 1000
    correct_paids_nr = np.array([0.9 ** (2 / 4)] * 2 + [1])

    assert np.all(np.isclose(paid_losses, correct_paids, rtol=TOLERANCE))
    assert np.all(np.isclose(reported_losses, correct_reported, rtol=TOLERANCE))
    assert np.all(np.isclose(earned_prem, correct_ep, rtol=TOLERANCE))
    assert np.all(np.isclose(paid_losses_lf, correct_paids, rtol=TOLERANCE))
    assert np.all(
        np.isclose(
            correct_weights_evaluation, derived_weights_evaluation, rtol=TOLERANCE
        )
    )
    assert np.all(
        np.isclose(
            reported_losses_lf,
            [cel["reported_loss"] for cel in test_triangle],
            rtol=TOLERANCE,
        )
    )
    assert np.all(np.isclose(paid_losses_nr, correct_paids_nr, rtol=TOLERANCE))


def test_validation():
    # Testing invalid decay factor
    with pytest.raises(ValueError):
        weight_geometric_decay(test_triangle, -0.9)

    # Testing invalid loss_field
    with pytest.raises(ValueError):
        weight_geometric_decay(
            test_triangle, 0.3, basis="experience", tri_fields="financial_loss"
        )
