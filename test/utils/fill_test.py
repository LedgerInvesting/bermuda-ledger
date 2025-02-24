import datetime

import bermuda as tri


def test_backfill():
    partial_tri = tri.binary_to_triangle("test/test_data/missing_eval.trib")
    full_tri = tri.utils.fill_forward_gaps(partial_tri)
    full_tri_nones = tri.utils.fill_forward_gaps(partial_tri, fill_with_none=True)
    new_nones = full_tri_nones.filter(
        lambda cell: cell.evaluation_date == datetime.date(2023, 12, 31)
    )
    assert len(full_tri) > len(partial_tri)
    assert len(full_tri_nones) > len(partial_tri)
    assert full_tri.periods == partial_tri.periods
    assert full_tri.right_edge == partial_tri.right_edge
    assert full_tri_nones.periods == partial_tri.periods
    assert full_tri_nones.right_edge == partial_tri.right_edge
    assert [cell["paid_loss"] for cell in new_nones] == [None] * len(new_nones)
