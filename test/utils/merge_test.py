from itertools import product

import pytest
from numpy.testing import assert_array_equal

from bermuda import Triangle, join, loose_period_merge, merge, period_merge
from bermuda.utils.merge import _merge_cell_pair

from ..triangle_samples import *

tri1 = Triangle(raw_obs)
overwrite = Triangle(raw_obs[1:]).replace(values={"paid_loss": -1})
tri1_inc = Triangle(raw_incremental_obs)
overwrite_inc = Triangle(raw_incremental_obs[1:]).replace(values={"paid_loss": -1})


def test_merge():
    """Only matching cells have values overwritten"""
    overwritten = merge(tri1, overwrite)
    paid_loss = [cell["paid_loss"] for cell in overwritten]
    overwritten_inc = merge(tri1_inc, overwrite_inc)
    paid_loss_inc = [cell["paid_loss"] for cell in overwritten_inc]

    assert raw_obs[0] in overwritten
    assert paid_loss == [100] + [-1] * 5
    assert raw_incremental_obs[0] in overwritten_inc
    assert paid_loss_inc == [100] + [-1] * 5

    assert _merge_cell_pair(raw_obs[0], None) == raw_obs[0]
    assert _merge_cell_pair(None, raw_obs[0]) == raw_obs[0]


def test_period_merge():
    """Use right edge to replace all cells by period."""
    period_overwrite = overwrite.right_edge
    overwritten = period_merge(tri1, period_overwrite)
    paid_loss = [cell["paid_loss"] for cell in overwritten]
    period_overwrite_inc = overwrite_inc.right_edge
    overwritten_inc = period_merge(tri1_inc, period_overwrite_inc)
    paid_loss_inc = [cell["paid_loss"] for cell in overwritten_inc]

    assert paid_loss == [-1] * 6
    assert paid_loss_inc == [-1] * 6

    assert period_merge(tri1, Triangle([])) == tri1
    assert period_merge(Triangle([]), tri1) == Triangle([])

    # Check for cell type consistency
    with pytest.raises(ValueError):
        period_merge(tri1, overwrite_inc)


def test_period_merge_suffix():
    original_paid = [cell["paid_loss"] for cell in tri1]

    period_overwrite = overwrite.right_edge
    overwritten = period_merge(tri1, period_overwrite, suffix="_test")

    paid_loss = [cell["paid_loss"] for cell in overwritten]
    paid_loss_suffix = [cell["paid_loss_test"] for cell in overwritten]

    assert_array_equal(original_paid, paid_loss)
    assert_array_equal(paid_loss_suffix, [-1] * len(paid_loss_suffix))


def test_merge_on():
    """Even if the `overwrite` triangle has slightly different
    metadata, we can still merge by specifying which metadata
    attributes we want to join on."""
    policy_overwrite = Triangle(
        [
            cell.replace(metadata=lambda _: Metadata(risk_basis="Policy"))
            for cell in overwrite
        ]
    )
    overwritten = merge(
        tri1,
        policy_overwrite,
        on=["reinsurance_basis", "loss_definition", "country", "currency"],
    )
    paid_loss = [cell["paid_loss"] for cell in overwritten]

    assert paid_loss == [100] + [-1] * 5


def test_period_merge_exception():
    """Raise exception when tri2 has multiple cells per period."""
    with pytest.raises(ValueError):
        period_merge(tri1, overwrite)
    with pytest.raises(ValueError):
        period_merge(tri1_inc, overwrite_inc)


def test_join():
    # tri1 and tri2 each have one unique cell
    unique_tri2_cell = overwrite.cells[-1].replace(
        evaluation_date=datetime.date(2050, 12, 31)
    )
    tri2 = overwrite + Triangle([unique_tri2_cell])
    unique_inc_tri2_cell = overwrite_inc.cells[-1].replace(
        evaluation_date=datetime.date(2050, 12, 31)
    )
    tri2_inc = overwrite_inc + Triangle([unique_inc_tri2_cell])

    full = join(tri1, tri2)
    left = join(tri1, tri2, "left")
    right = join(tri1, tri2, "right")
    inner = join(tri1, tri2, "inner")  # noqa: F841
    left_anti = join(tri1, tri2, "left_anti")
    right_anti = join(tri1, tri2, "right_anti")
    full_inc = join(tri1_inc, tri2_inc)
    left_inc = join(tri1_inc, tri2_inc, "left")
    right_inc = join(tri1_inc, tri2_inc, "right")
    inner_inc = join(tri1_inc, tri2_inc, "inner")  # noqa: F841
    left_anti_inc = join(tri1_inc, tri2_inc, "left_anti")
    right_anti_inc = join(tri1_inc, tri2_inc, "right_anti")

    with pytest.raises(ValueError):
        join(tri1_inc, tri2)

    with pytest.raises(ValueError):
        join(tri1, tri2, "all")

    assert any([cell1 is None for cell1, cell2 in full])
    assert any([cell2 is None for cell1, cell2 in full])
    assert any([cell1 is None for cell1, cell2 in full_inc])
    assert any([cell2 is None for cell1, cell2 in full_inc])

    assert not (any([cell1 is None for cell1, cell2 in left]))
    assert any([cell2 is None for cell1, cell2 in left])
    assert not (any([cell1 is None for cell1, cell2 in left_inc]))
    assert any([cell2 is None for cell1, cell2 in left_inc])

    assert any([cell1 is None for cell1, cell2 in right])
    assert not (any([cell2 is None for cell1, cell2 in right]))
    assert any([cell1 is None for cell1, cell2 in right_inc])
    assert not (any([cell2 is None for cell1, cell2 in right_inc]))

    assert not (any([cell1 is None for cell1, cell2 in left_anti]))
    assert all([cell2 is None for cell1, cell2 in left_anti])
    assert not (any([cell1 is None for cell1, cell2 in left_anti_inc]))
    assert all([cell2 is None for cell1, cell2 in left_anti_inc])

    assert all([cell1 is None for cell1, cell2 in right_anti])
    assert not (any([cell2 is None for cell1, cell2 in right_anti]))
    assert all([cell1 is None for cell1, cell2 in right_anti_inc])
    assert not (any([cell2 is None for cell1, cell2 in right_anti_inc]))


def test_loose_period_merge():
    """Use right edge to replace all cells by period."""
    multi_slice_tri = sum(
        tri1.derive_metadata(details={"state": state, "coverage": coverage})
        for state, coverage in product(["NY", "NJ"], ["comprehensive", "collision"])
    )
    period_overwrite = sum(
        overwrite.right_edge.derive_metadata(details={"state": state}).derive_fields(
            paid_loss=paid
        )
        for state, paid in zip(["NY", "NJ"], [-1, -2])
    )

    overwritten = loose_period_merge(multi_slice_tri, period_overwrite)
    assert overwritten.metadata == multi_slice_tri.metadata
    assert [
        cell["paid_loss"]
        for cell in overwritten.filter(lambda cell: cell.details["state"] == "NY")
    ] == [-1] * 12
    assert [
        cell["paid_loss"]
        for cell in overwritten.filter(lambda cell: cell.details["state"] == "NJ")
    ] == [-2] * 12

    with pytest.raises(ValueError):
        loose_period_merge(period_overwrite, multi_slice_tri)
