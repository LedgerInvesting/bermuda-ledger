import datetime
import functools

import numpy as np
import pytest

from bermuda import (
    Cell,
    IncrementalCell,
    Metadata,
    Triangle,
    TriangleError,
    blend,
    summarize,
)
from bermuda.utils.summarize import _metadata_attr_gcd, _metadata_gcd


@functools.cache
def build_test_triangles():
    np.random.seed(1234)
    years = [2021, 2022, 2023, 2024]
    metadata_sets = [
        Metadata(details={"program_tag": "ABC.DEF"}),
        Metadata(details={"program_tag": "BCD.DEF"}),
    ]

    base_tri = Triangle(
        [
            Cell(
                period_start=datetime.date(exp_year, 1, 1),
                period_end=datetime.date(exp_year, 12, 31),
                evaluation_date=datetime.date(eval_year, 12, 31),
                values=dict(
                    reported_loss=np.random.normal(size=1000, loc=1e6, scale=1e5),
                    earned_premium=1e6,
                ),
                metadata=metadata,
            )
            for metadata in metadata_sets
            for exp_year in years
            for eval_year in years
            if eval_year >= exp_year
        ]
    )

    alt_tri = base_tri.replace(
        values=lambda cell: dict(
            earned_premium=cell["earned_premium"],
            reported_loss=np.random.normal(size=1000, loc=8e5, scale=1e5),
        )
    )

    inc_tri = Triangle(
        [
            IncrementalCell(
                period_start=datetime.date(exp_year, 1, 1),
                period_end=datetime.date(exp_year, 12, 31),
                evaluation_date=datetime.date(eval_year, 12, 31),
                prev_evaluation_date=datetime.date(eval_year - 1, 12, 31),
                values=dict(
                    reported_loss=np.random.normal(size=1000, loc=1e6, scale=1e5),
                    earned_premium=1e6,
                ),
                metadata=metadata,
            )
            for metadata in metadata_sets
            for exp_year in years
            for eval_year in years
            if eval_year >= exp_year
        ]
    )

    alt_inc_tri = inc_tri.replace(
        values=lambda cell: dict(
            earned_premium=cell["earned_premium"],
            reported_loss=np.random.normal(size=1000, loc=8e5, scale=1e5),
        )
    )

    return base_tri, alt_tri, inc_tri, alt_inc_tri


simple_triangle = Triangle(
    [
        Cell(
            period_start=datetime.date(2023, 1, 1),
            period_end=datetime.date(2023, 12, 31),
            evaluation_date=datetime.date(2023, 12, 31),
            values={
                "reported_loss": np.array([1, 2, 3]),
                "earned_premium": 1,
            },
        )
    ]
)


def test_simple_blend_mixture():
    simple_triangle_flip = simple_triangle.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": 1,
        }
    )
    blended_tri = simple_triangle.blend(triangles=[simple_triangle_flip], seed=1234)

    values = blended_tri[0]["reported_loss"]
    assert len(values) == 3
    assert all(values == [1, 2, 3])


def test_simple_blend_linear():
    simple_triangle_flip = simple_triangle.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": 3,
        }
    )
    simple_triangle_arrays = simple_triangle_flip.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": np.array([2, 3, 4]),
        }
    )
    blended_tri_flip = simple_triangle.blend([simple_triangle_flip], method="linear")
    blended_tri_array = simple_triangle.blend([simple_triangle_arrays], method="linear")

    values_flip = blended_tri_flip[0].values
    values_array = blended_tri_array[0].values

    assert all(values_flip["earned_premium"] == 2)
    assert all(values_flip["reported_loss"] == [2, 2, 2])
    assert all(values_array["reported_loss"] == [2, 2, 2])
    assert all(values_array["earned_premium"] == [1.5, 2, 2.5])

    with pytest.raises(ValueError):
        simple_triangle.select(["reported_loss"]).blend(
            [simple_triangle],
            method="linear",
        )


def test_simple_blend_mixture_cellwise_weights():
    simple_triangle_flip = simple_triangle.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": 1,
        }
    )
    blended_tri = blend(
        [simple_triangle, simple_triangle_flip],
        weights={"tri1": [0.5], "tri2": [0.5]},
        seed=1234,
    )

    values = blended_tri[0]["reported_loss"]
    assert len(values) == 3
    assert all(values == [1, 2, 3])


def test_simple_blend_linear_cellwise_weights():
    simple_triangle_flip = simple_triangle.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": 2,
        }
    )
    blended_tri = blend(
        [simple_triangle, simple_triangle_flip],
        weights={"tri1": [0.5], "tri2": [0.5]},
        method="linear",
    )

    values = blended_tri[0].values
    assert len(values["reported_loss"]) == 3
    assert len(values["earned_premium"]) == 1
    assert all(values["reported_loss"] == [2, 2, 2])
    assert values["earned_premium"] == 1.5


def test_blend_errors_raised():
    simple_triangle_flip = simple_triangle.replace(
        values={
            "reported_loss": np.array([3, 2, 1]),
            "earned_premium": 1,
        }
    )

    with pytest.raises(ValueError):
        blend(
            [simple_triangle.select(["reported_loss"]), simple_triangle],
            method="mixture",
        )

    with pytest.raises(ValueError):
        # Different scalars
        blend(
            [simple_triangle.derive_fields(earned_premium=2), simple_triangle],
            method="mixture",
        )

    with pytest.raises(TypeError):
        # Different types
        blend(
            [
                simple_triangle.derive_fields(earned_premium=np.array([2, 2])),
                simple_triangle,
            ],
            method="mixture",
        )

    with pytest.raises(TypeError):
        # Can only blend lists
        blend((simple_triangle, simple_triangle_flip))

    with pytest.raises(TypeError):
        # Weights need to be lists or dictionaries
        blend([simple_triangle, simple_triangle_flip], weights=(0.5, 0.5))

    with pytest.raises(ValueError):
        # Blending single triangles require weights=1.0
        blend([simple_triangle], weights=[0.5])


def test_summarization_blend():
    base_tri, alt_tri, inc_tri, alt_inc_tri = build_test_triangles()
    blend_tri = blend([base_tri, alt_tri], seed=1234)
    blend_inc_tri = blend([inc_tri, alt_inc_tri], seed=1234)

    assert len(blend_tri) == len(base_tri)
    assert len(blend_tri.slices) == len(base_tri.slices)
    assert blend_tri.cells[0]["reported_loss"].shape[0] == 1000
    assert np.round(blend_tri.cells[0]["reported_loss"].mean(), -4) == 900e3
    assert blend_tri.cells[0]["earned_premium"] == 1e6

    assert len(blend_inc_tri) == len(inc_tri)
    assert len(blend_inc_tri.slices) == len(inc_tri.slices)
    assert blend_inc_tri.cells[0]["reported_loss"].shape[0] == 1000
    assert np.round(blend_inc_tri.cells[0]["reported_loss"].mean(), -4) == 900e3
    assert blend_inc_tri.cells[0]["earned_premium"] == 1e6


def test_summarization_reweight():
    base_tri, alt_tri, inc_tri, alt_inc_tri = build_test_triangles()
    blend_tri = blend([base_tri, alt_tri], weights=[0.99, 0.01], seed=1234)
    blend_inc_tri = blend([inc_tri, alt_inc_tri], weights=[0.99, 0.01], seed=1234)
    # If almost all of the weight is on the first triangle,
    # we should be closer to the value in the first estimate
    assert np.round(blend_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3
    assert np.round(blend_inc_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3

    with pytest.raises(ValueError):
        blend([base_tri, alt_tri], weights=[0.2, 0.3, 0.5], seed=1234)


def test_summarization_reweight_dict_array():
    base_tri, alt_tri, inc_tri, alt_inc_tri = build_test_triangles()
    cellwise_weights = {
        "tri1": np.atleast_2d([0.99]),
        "tri2": np.atleast_2d([0.01]),
    }
    blend_tri = blend([base_tri, alt_tri], weights=cellwise_weights, seed=1234)
    blend_inc_tri = blend([inc_tri, alt_inc_tri], weights=cellwise_weights, seed=1234)
    # If almost all of the weight is on the first triangle,
    # we should be closer to the value in the first estimate
    assert np.round(blend_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3
    assert np.round(blend_inc_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3

    # should fail if given more weights than triangles
    cellwise_weights["tri3"] = np.atleast_2d([0])
    with pytest.raises(ValueError):
        blend([base_tri, alt_tri], weights=cellwise_weights, seed=1234)


def test_summarization_reweight_cellwise():
    base_tri, alt_tri, inc_tri, alt_inc_tri = build_test_triangles()
    cellwise_weights = {
        "tri1": np.atleast_2d(np.linspace(1, 0, len(base_tri))),
        "tri2": np.atleast_2d(np.linspace(0, 1, len(base_tri))),
    }
    blend_tri = blend([base_tri, alt_tri], weights=cellwise_weights, seed=1234)
    blend_inc_tri = blend([inc_tri, alt_inc_tri], weights=cellwise_weights, seed=1234)

    # If almost all of the weight is on the first cell of the first triangle,
    # we should be closer to its value in the first blended cell
    assert np.round(blend_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3
    assert np.round(blend_inc_tri.cells[0]["reported_loss"].mean(), -4) == 1000e3

    # as weight increases to the second triangle for the later cells,
    # we should be closter to the second trial for the last blended cell
    assert np.round(blend_tri.cells[-1]["reported_loss"].mean(), -4) == 8000e2
    assert np.round(blend_inc_tri.cells[-1]["reported_loss"].mean(), -4) == 8000e2

    # should fail if given more weights than triangles
    cellwise_weights["tri3"] = np.atleast_2d(np.zeros(len(base_tri)))
    with pytest.raises(ValueError):
        blend([base_tri, alt_tri], weights=cellwise_weights, seed=1234)


def test_blend_0_1():
    base_tri, alt_tri, _, _ = build_test_triangles()
    blend_tri = blend([base_tri, alt_tri], weights=[1, 0], seed=1234)
    # If  all of the weight is on the first triangle it should equal the first triangle
    assert blend_tri == base_tri

    with pytest.raises(ValueError):
        blend([base_tri, alt_tri], weights=[0.2, 0.3, 0.5], seed=1234)


def test_blend_1():
    base_tri, _, _, _ = build_test_triangles()
    blend_tri = blend([base_tri], weights=[1], seed=1234)
    # If  all of the weight is on the first triangle it should equal the first triangle
    assert blend_tri == base_tri


def test_summarization_checks():
    base_tri, alt_tri, inc_tri, _ = build_test_triangles()

    # What if one triangle is missing a whole slice?
    with pytest.raises(ValueError):
        blend([list(base_tri.slices.values())[0], alt_tri])

    # What if one triangle is missing a diagonal?
    with pytest.raises(ValueError):
        blend([base_tri.clip(max_eval=datetime.date(2023, 12, 31)), alt_tri])

    # What if one triangle doesn't have a crucial field?
    with pytest.raises(ValueError):
        blend([base_tri, alt_tri.select(["earned_premium"])])

    # What if scalars don't match?
    with pytest.raises(ValueError):
        blend([base_tri, alt_tri.derive_fields(earned_premium=lambda cell: 12345.67)])

    # What if data types don't match?
    with pytest.raises(TypeError):
        blend([base_tri.derive_fields(reported_loss=lambda cell: 123_456), alt_tri])

    # What if coordinates don't match?
    with pytest.raises(ValueError):
        blend(
            [
                base_tri.replace(metadata=Metadata(details={"program_tag": "XYZ.ABC"})),
                alt_tri,
            ]
        )

    # Check cell type consistency
    with pytest.raises(ValueError):
        blend([base_tri, inc_tri])


def test_summarization_metadata():
    base_tri, _, _, _ = build_test_triangles()
    assert (
        _metadata_attr_gcd(
            Triangle(
                [base_tri.cells[0].derive_metadata(country="USA")] + base_tri.cells[1:]
            ),
            "country",
        )
        is None
    )
    # Check risk basis consistency
    with pytest.raises(TriangleError):
        _metadata_gcd(
            Triangle(
                [base_tri.cells[0].derive_metadata(risk_basis="Policy")]
                + base_tri.cells[1:]
            )
        )
    # Check currency consistency
    with pytest.raises(TriangleError):
        _metadata_gcd(
            Triangle(
                [base_tri.cells[0].derive_metadata(currency="CAD")] + base_tri.cells[1:]
            )
        )
