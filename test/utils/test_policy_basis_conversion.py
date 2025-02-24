from datetime import date

import numpy as np
import pytest

import bermuda as tri


@pytest.fixture
def ragged_triangle():
    return tri.binary_to_triangle("test/test_data/ragged_aq_triangle.trib")


def test_accident_quarter_to_policy(ragged_triangle):
    aq_tri = ragged_triangle.clip(min_period=date(2017, 10, 1))
    py_tri = tri.accident_quarter_to_policy_year(
        aq_tri.clip(min_period=date(2017, 10, 1))
    )

    aq_ep = sum(cell["earned_premium"] for cell in aq_tri.right_edge)
    py_ep = sum(cell["earned_premium"] for cell in py_tri.right_edge)

    aq_loss = sum(cell["paid_loss"] for cell in aq_tri.right_edge)
    py_loss = sum(cell["paid_loss"] for cell in py_tri.right_edge)

    assert np.isclose(aq_ep, py_ep)
    assert np.isclose(aq_loss, py_loss)
    assert py_tri.evaluation_date == aq_tri.evaluation_date
    assert py_tri[0].metadata.risk_basis == "Policy"


def test_aq_to_py_ragged_error(ragged_triangle):
    # test that it raises an error when called on the mis-shapen triangle
    with pytest.raises(ValueError):
        tri.utils.accident_quarter_to_policy_year(ragged_triangle)
