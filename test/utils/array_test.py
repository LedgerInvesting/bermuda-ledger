import pytest
from numpy.testing import assert_array_equal

from bermuda import Triangle, TriangleError, array_from_field

from ..triangle_samples import array_raw_obs, raw_obs


@pytest.fixture
def triangle_scalar():
    return Triangle(raw_obs)


@pytest.fixture
def triangle_vector():
    return Triangle(array_raw_obs)


def test_array_from_field(triangle_scalar):
    paid_loss = array_from_field(triangle_scalar, "paid_loss")
    assert_array_equal(paid_loss, [ob["paid_loss"] for ob in raw_obs])

    reported_loss = array_from_field(triangle_scalar, "reported_loss")
    assert_array_equal(reported_loss, [ob["reported_loss"] for ob in raw_obs])


def test_array_from_field_vector(triangle_vector):
    paid_loss = array_from_field(triangle_vector, "paid_loss")
    assert_array_equal(
        paid_loss,
        [loss * 10 for loss in [[100], [160], [190], [150], [220], [120]]],
    )


def test_array_from_field_non_value_field(triangle_scalar):
    with pytest.raises(TriangleError):
        array_from_field(triangle_scalar, "period_start")
