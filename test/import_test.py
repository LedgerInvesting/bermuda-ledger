import subprocess
import sys

import bermuda


def test_meyers_triangle_is_loaded_lazily():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import bermuda; "
                "assert 'meyers_tri' not in vars(bermuda); "
                "tri = bermuda.meyers_tri; "
                "assert 'meyers_tri' in vars(bermuda); "
                "assert len(tri) > 0"
            ),
        ],
        check=True,
    )
    assert result.returncode == 0


def test_root_exports_remain_available():
    assert bermuda.Triangle is not None
    assert bermuda.Metadata is not None
    assert bermuda.json_to_triangle is not None
    assert bermuda.aggregate is not None
    assert bermuda.Matrix is not None


def test_lazy_bindings_execute():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import bermuda; "
                "tri = bermuda.meyers_tri; "
                "payload = tri.to_dict(); "
                "round_trip = bermuda.Triangle.from_dict(payload); "
                "assert round_trip == tri"
            ),
        ],
        check=True,
    )
    assert result.returncode == 0
