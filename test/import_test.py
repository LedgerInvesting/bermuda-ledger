import bermuda


def test_meyers_triangle_is_loaded_lazily():
    assert "meyers_tri" not in vars(bermuda)

    meyers_tri = bermuda.meyers_tri

    assert "meyers_tri" in vars(bermuda)
    assert len(meyers_tri) == 100


def test_root_exports_remain_available():
    assert bermuda.Triangle is not None
    assert bermuda.Metadata is not None
    assert bermuda.json_to_triangle is not None
    assert bermuda.aggregate is not None
    assert bermuda.Matrix is not None
