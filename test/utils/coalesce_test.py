from datetime import date

import pytest

import bermuda as tri

from ..triangle_samples import raw_obs

tri1 = (
    tri.Triangle(raw_obs)
    .clip(max_period=date(2020, 4, 1))
    .derive_fields(paid_loss=lambda _: 1)
)
tri2 = (
    tri.Triangle(raw_obs)
    .clip(max_period=date(2020, 7, 1))
    .derive_fields(paid_loss=lambda _: 2)
)
tri3 = tri.Triangle(raw_obs).derive_fields(paid_loss=lambda _: 3)


def test_coalesce():
    combined = tri.coalesce([tri1, tri2, tri3])
    assert combined.clip(max_period=date(2020, 4, 1)) == tri1
    assert all(
        cell["paid_loss"] == 2
        for cell in combined.filter(lambda cell: cell.period_start == date(2020, 4, 1))
    )
    assert all(
        cell["paid_loss"] == 3
        for cell in combined.filter(lambda cell: cell.period_start == date(2020, 7, 1))
    )


def test_no_matching():
    with pytest.warns(UserWarning):
        combined = tri.coalesce([tri1])

    assert combined == tri1
