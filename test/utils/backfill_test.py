import bermuda as tri


def test_backfill():
    partial_tri = tri.binary_to_triangle("test/test_data/missing_cells.trib")
    full_tri = tri.utils.backfill(partial_tri)
    assert full_tri[0].dev_lag() == 0
    assert full_tri[0]["paid_loss"] == 0
    assert full_tri[0]["reported_loss"] == 0
    assert full_tri[0]["earned_premium"] > 0
