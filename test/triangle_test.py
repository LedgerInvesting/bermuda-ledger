from datetime import date

import numpy as np
import pytest
from deepdiff import DeepDiff

from bermuda import Triangle, TriangleSlice
from bermuda.base import Cell, IncrementalCell, Metadata
from bermuda.errors import *
from bermuda.utils import slice_to_triangle, triangle_to_slice

from .triangle_samples import *

base_tri = Triangle(raw_obs)
ordered_tri = Triangle(disordered_obs)
incremental_tri = Triangle(raw_incremental_obs)
cumulative_tri = Triangle(raw_cumulative_obs)
period_agg_tri = Triangle(period_agg_obs)
period_agg_inc_tri = Triangle(period_agg_inc_obs)
eval_agg_tri = Triangle([raw_obs[0], raw_obs[2]] + raw_obs[4:])
eval_agg_inc_tri = Triangle(eval_agg_inc_obs)
full_agg_tri = Triangle([period_agg_obs[0]] + period_agg_obs[2:])
trivial_tri = Triangle([raw_obs[1], raw_obs[3], raw_obs[5]])
gapped_tri = Triangle(raw_obs[:4] + raw_obs[-1:])
short_tri = Triangle(raw_obs[:2] + [raw_obs[3]])
premium_tri = Triangle(premium_obs)
experience_gap_tri = Triangle(experience_gap_obs)
ten_samples = Triangle(array_raw_obs)

two_slice_tri = base_tri + Triangle(
    [
        Cell(
            period_start=cell.period_start,
            period_end=cell.period_end,
            evaluation_date=cell.evaluation_date,
            metadata=Metadata(risk_basis="Policy"),
            values=cell.values,
        )
        for cell in raw_obs
    ]
)


def test_accessors():
    assert list(base_tri.slices.values()) == [base_tri]


def test_index():
    assert base_tri[0] == base_tri.cells[0]


def test_slice_index():
    tri_slice = base_tri[:2]
    assert isinstance(tri_slice, Triangle)
    assert base_tri[:2].cells == base_tri.cells[:2]


def test_indices():
    single_cell = base_tri[date(2020, 1, 1), date(2020, 3, 31), :]
    single_row = base_tri[date(2020, 1, 1), :, :]
    whole_triangle = base_tri[
        date(2020, 1, 1) : date(2020, 7, 1), date(2020, 3, 31) : date(2020, 9, 30), :
    ]
    cell_only = two_slice_tri[
        date(2020, 1, 1), date(2020, 3, 31), Metadata(risk_basis="Policy")
    ]

    assert isinstance(single_cell, Triangle)
    assert single_cell.cells[0].period_start == date(2020, 1, 1)
    assert single_cell.cells[0].evaluation_date == date(2020, 3, 31)

    assert isinstance(single_row, Triangle)
    assert single_row.periods[0] == (date(2020, 1, 1), date(2020, 3, 31))

    assert whole_triangle == base_tri

    assert isinstance(cell_only, Cell)

    # slice(0) will give None `start` and 0 `stop`
    res = base_tri[slice(0), slice(None), None]
    assert not DeepDiff(res, base_tri)


def test_triangle_slice():
    tri_slice = triangle_to_slice(base_tri)
    original_triangle = slice_to_triangle(tri_slice)

    assert isinstance(tri_slice, TriangleSlice)
    assert isinstance(original_triangle, Triangle)
    assert base_tri == original_triangle


def test_triangle_slice_error():
    with pytest.raises(TriangleError):
        triangle_to_slice(two_slice_tri)


def test_triangle_slice_indexing():
    tri_slice = triangle_to_slice(base_tri)

    first_cell = tri_slice[0]
    first_cells = tri_slice[:2]

    period_slice = tri_slice[date(2020, 1, 1), :]
    single_cell = tri_slice[date(2020, 1, 1), date(2020, 3, 31)]

    assert isinstance(first_cell, Cell)
    assert isinstance(single_cell, Cell)
    assert isinstance(first_cells, TriangleSlice)
    assert isinstance(period_slice, TriangleSlice)


def test_simple_properties():
    assert len(base_tri) == 6

    assert base_tri.periods == [
        (date(2020, 1, 1), date(2020, 3, 31)),
        (date(2020, 4, 1), date(2020, 6, 30)),
        (date(2020, 7, 1), date(2020, 9, 30)),
    ]

    assert base_tri.dev_lags("months") == [0.0, 3.0, 6.0]

    assert base_tri.evaluation_dates == [
        date(2020, 3, 31),
        date(2020, 6, 30),
        date(2020, 9, 30),
    ]
    assert base_tri.evaluation_date == date(2020, 9, 30)

    assert base_tri.fields == ["paid_loss", "reported_loss"]
    assert not base_tri.is_empty


def test_experience_gaps_property():
    experience_gaps = experience_gap_tri.experience_gaps
    expected_gaps = [
        (datetime.date(2019, 4, 1), datetime.date(2019, 9, 30)),
        (datetime.date(2020, 1, 1), datetime.date(2020, 6, 30)),
    ]
    assert experience_gaps == expected_gaps


def test_cells():
    assert base_tri.cells == raw_obs


def test_cell_type_consistency():
    with pytest.raises(TriangleError):
        Triangle([raw_obs[0], raw_incremental_obs[0]])

    with pytest.raises(TriangleError):
        Triangle([raw_obs[0], raw_cumulative_obs[0]])

    with pytest.raises(TriangleError):
        Triangle([raw_cumulative_obs[0], raw_incremental_obs[0]])


def test_sorted_cells():
    meta_details = [cell.details["a"] for cell in ordered_tri]
    assert ordered_tri.cells == sorted(disordered_obs)
    assert meta_details == [1, 1, 2]


def test_triangle_coordinates():
    with pytest.warns(DuplicateCellWarning):
        Triangle(
            [
                Cell(
                    period_start=datetime.date(1980, 1, 1),
                    period_end=datetime.date(1980, 12, 31),
                    evaluation_date=datetime.date(1980, 12, 31),
                    values={"paid_loss": 0},
                ),
                Cell(
                    period_start=datetime.date(1980, 1, 1),
                    period_end=datetime.date(1980, 12, 31),
                    evaluation_date=datetime.date(1980, 12, 31),
                    values={"paid_loss": 10},
                ),
            ]
        )
    with pytest.warns(DuplicateCellWarning):
        Triangle(
            [
                IncrementalCell(
                    period_start=datetime.date(1980, 1, 1),
                    period_end=datetime.date(1980, 12, 31),
                    evaluation_date=datetime.date(1980, 12, 31),
                    prev_evaluation_date=datetime.date(1978, 12, 31),
                    values={"paid_loss": 0},
                ),
                IncrementalCell(
                    period_start=datetime.date(1980, 1, 1),
                    period_end=datetime.date(1980, 12, 31),
                    evaluation_date=datetime.date(1980, 12, 31),
                    prev_evaluation_date=datetime.date(1979, 12, 31),
                    values={"paid_loss": 10},
                ),
            ]
        )


def test_categorization():
    assert base_tri.is_disjoint
    assert base_tri.is_semi_regular(dev_lag_unit="month")
    assert base_tri.is_regular(dev_lag_unit="month")
    assert base_tri.has_consistent_currency
    assert base_tri.has_consistent_risk_basis


def test_empty_categorization():
    empty_tri = Triangle([])
    assert empty_tri.is_disjoint
    assert empty_tri.is_semi_regular(dev_lag_unit="month")
    assert empty_tri.is_regular(dev_lag_unit="month")


def test_filter():
    filter_tri = base_tri.filter(lambda cell: cell["paid_loss"] > 150)
    assert len(filter_tri) == 3
    assert min([cell["paid_loss"] for cell in filter_tri]) > 150

    filter_inc_tri = incremental_tri.filter(lambda cell: cell["reported_loss"] > 300)
    assert len(filter_inc_tri) == 2
    assert min([cell["reported_loss"] for cell in filter_inc_tri]) > 300

    empty_tri = base_tri.filter(lambda cell: cell.values.get("selifjsef", 1000) < 140)
    assert empty_tri.is_empty


def test_derive_field():
    derived_tri = base_tri.derive_fields(
        case_reserve=lambda cell: cell["reported_loss"] - cell["paid_loss"],
    )
    assert [cell["case_reserve"] for cell in derived_tri.cells] == [
        100,
        110,
        120,
        250,
        260,
        380,
    ]

    derived_inc_tri = incremental_tri.derive_fields(
        earned_premium=lambda cell: cell["reported_loss"] * 1.1,
    )
    assert [np.round(cell["earned_premium"], 5) for cell in derived_inc_tri.cells] == [
        220.0,
        77.0,
        44.0,
        440.0,
        88.0,
        550.0,
    ]


def test_derive_metadata():
    derived_tri = base_tri.derive_metadata(currency="USD")
    assert all([cell.metadata.currency == "USD" for cell in derived_tri])

    derived_inc_tri = incremental_tri.derive_metadata(currency="USD")
    assert all([cell.metadata.currency == "USD" for cell in derived_inc_tri])


def test_date_clipping():
    clip_period_tri = base_tri.clip(min_period=date(2020, 4, 1))
    assert len(clip_period_tri) == 3
    assert clip_period_tri.periods == [
        (date(2020, 4, 1), date(2020, 6, 30)),
        (date(2020, 7, 1), date(2020, 9, 30)),
    ]
    clip_period_max_tri = base_tri.clip(max_period=date(2020, 7, 1))
    assert len(clip_period_max_tri) == 5
    assert clip_period_max_tri.periods == [
        (date(2020, 1, 1), date(2020, 3, 31)),
        (date(2020, 4, 1), date(2020, 6, 30)),
    ]
    assert clip_period_max_tri.dev_lags("month") == [0.0, 3.0, 6.0]

    clip_eval_tri = base_tri.clip(max_eval=date(2020, 8, 15))
    assert clip_eval_tri == short_tri

    clip_eval_tri = base_tri.clip(min_eval=date(2020, 2, 15))
    assert clip_eval_tri == base_tri

    clip_eval_tri = base_tri.clip(min_eval=date(2020, 4, 1))
    assert len(clip_eval_tri) == 5
    assert clip_eval_tri.periods == base_tri.periods
    assert clip_eval_tri.dev_lags("month") == base_tri.dev_lags("month")
    assert clip_eval_tri.evaluation_dates == [date(2020, 6, 30), date(2020, 9, 30)]


def test_right_edge():
    assert trivial_tri.right_edge == trivial_tri


def test_is_right_edge_ragged():
    assert trivial_tri.is_right_edge_ragged
    assert not period_agg_tri.is_right_edge_ragged


def test_to_frame():
    frame = base_tri.to_data_frame()
    frame_inc = incremental_tri.to_data_frame()
    assert len(frame) == 6
    assert len(frame.columns) == 12
    assert len(frame_inc.columns) == 13
    assert frame["period_start"].tolist()[0] == date(2020, 1, 1)
    assert frame["period_start"].tolist()[-1] == date(2020, 7, 1)
    assert frame_inc["period_start"].tolist()[0] == date(2020, 1, 1)
    assert frame_inc["period_start"].tolist()[-1] == date(2020, 7, 1)


def test_repr_html():
    # noinspection PyUnusedLocal
    res = base_tri._repr_html_()  # noqa: F841
    assert True


def test_num_samples():
    ragged = Triangle(
        [
            cell.replace(values=lambda cell: {"paid_loss": np.full(size, 10)})
            for cell, size in zip(ten_samples, range(10, 14))
        ]
    )
    assert base_tri.num_samples == 1
    assert ten_samples.num_samples == 10
    with pytest.raises(ValueError):
        ragged.num_samples


def test_triangle_constructor_non_cell_values():
    with pytest.raises(TriangleError, match="can only hold `Cell`"):
        Triangle(
            [
                *raw_obs,
                ["some", "other", "data", "type"],
            ]
        )


def test_triangle_getter_invalid_index():
    with pytest.raises(ValueError, match="Must pass three indices"):
        base_tri[1, 2]

    with pytest.raises(ValueError, match="period_slice must be date or slice"):
        base_tri[0, 1, 2]

    with pytest.raises(ValueError, match="evaluation_slice must be date"):
        base_tri[slice(0), 1, None]


def test_triangle_common_and_diff_metadata():
    # single cell of metadata
    cell_w_meta = Cell(
        period_start=date(2020, 1, 1),
        period_end=date(2020, 3, 31),
        evaluation_date=date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
        metadata=Metadata(
            country="CA",
            currency="CAD",
            details={"shared_1": 1, "diff_value": 0, "diff_key_1": 1},
        ),
    )
    single_cell_tri = Triangle([cell_w_meta])
    assert single_cell_tri.common_metadata == cell_w_meta.metadata

    # multiple cells
    multi_cell_tri = Triangle(
        [
            cell_w_meta,
            Cell(
                period_start=date(2020, 1, 1),
                period_end=date(2020, 3, 31),
                evaluation_date=date(2020, 3, 31),
                values={"paid_loss": 100, "reported_loss": 200},
                metadata=Metadata(
                    country="CA",
                    currency="CAD",
                    details={"shared_1": 1, "diff_value": 1, "diff_key_2": 2},
                ),
            ),
        ]
    )
    expected_common_meta = Metadata(
        country="CA",
        currency="CAD",
        details={"shared_1": 1},
    )
    assert not DeepDiff(multi_cell_tri.common_metadata, expected_common_meta)

    expected_meta_diff = [
        Metadata(risk_basis=None, details={"diff_value": 0, "diff_key_1": 1}),
        Metadata(risk_basis=None, details={"diff_value": 1, "diff_key_2": 2}),
    ]
    assert not DeepDiff(multi_cell_tri.metadata_differences, expected_meta_diff)


def test_constant_details_removal():
    common_details = Triangle(
        [
            Cell(
                period_start=date(2020, 1, 1),
                period_end=date(2020, 3, 31),
                evaluation_date=date(2020, 3, 31),
                values={"paid_loss": 100, "reported_loss": 200},
                metadata=Metadata(
                    details={"common": 1, "different": different},
                    loss_details={"same": 1, "diff": different + 1},
                ),
            )
            for different in [1, 2, 3]
        ]
    )
    no_common_details = common_details.remove_static_details()
    empty_tri = Triangle([]).remove_static_details()

    assert list(no_common_details.metadata[0].details.keys()) == ["different"]
    assert list(no_common_details.metadata[0].loss_details.keys()) == ["diff"]
    assert [cell.metadata.details["different"] for cell in no_common_details] == [
        1,
        2,
        3,
    ]
    assert [cell.metadata.loss_details["diff"] for cell in no_common_details] == [
        2,
        3,
        4,
    ]
    assert empty_tri == Triangle([])


def test_extract():
    base_tri_prem = base_tri.derive_fields(earned_premium=1e3)
    paid_losses = base_tri.extract("paid_loss").sum()
    paid_losses_func = base_tri.extract(lambda cell: cell["paid_loss"]).sum()

    assert (
        paid_losses == paid_losses_func == sum(cell["paid_loss"] for cell in base_tri)
    )
    assert (
        base_tri_prem.extract("paid_loss").sum()
        / base_tri_prem.extract("earned_premium").sum()
    ).round(1) == 0.2


def test_aggregate():
    aggregated_periods = base_tri.aggregate(period_resolution=(1, "year"))
    aggregated_evals = base_tri.aggregate(
        eval_resolution=(1, "years"), eval_origin=datetime.date(2020, 3, 31)
    )

    assert len(aggregated_periods.periods) == 1
    assert len(aggregated_evals) == 1
