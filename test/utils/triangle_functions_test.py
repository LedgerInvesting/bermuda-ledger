import datetime
from itertools import product

import toolz as tlz
from numpy.testing import assert_array_almost_equal

from bermuda import *

from ..triangle_test import *


def test_aggregate():
    assert period_agg_tri == aggregate(base_tri, period_resolution=(6, "months"))
    assert eval_agg_tri == aggregate(
        base_tri,
        eval_resolution=(6, "months"),
        eval_origin=datetime.date(2020, 3, 31),
    )
    assert full_agg_tri == aggregate(
        base_tri,
        period_resolution=(6, "months"),
        eval_resolution=(6, "months"),
        eval_origin=datetime.date(2020, 3, 31),
    )
    assert period_agg_inc_tri == aggregate(
        incremental_tri, period_resolution=(6, "months")
    )
    assert eval_agg_inc_tri == aggregate(
        incremental_tri,
        eval_resolution=(6, "months"),
        eval_origin=datetime.date(2020, 3, 31),
    )
    # check if triangle periods cross aggregate period bounds
    with pytest.raises(TriangleError):
        aggregate(
            base_tri,
            period_resolution=(6, "month"),
            period_origin=datetime.date(2020, 1, 31),
        )


def test_disaggregate():
    dis_array_tri = disaggregate(
        ten_samples, resolution_exp_months=1, resolution_dev_months=1
    )
    agg_array_tri = aggregate(
        dis_array_tri, period_resolution=(3, "month"), eval_resolution=(3, "month")
    )
    for field in base_tri.fields:
        field_vals = array_from_field(ten_samples, field)
        agg_field_vals = array_from_field(agg_array_tri, field)
        assert_array_almost_equal(field_vals, agg_field_vals, decimal=13)

    with pytest.raises(TriangleError):
        disaggregate(Triangle(period_agg_obs))

    with pytest.raises(ValueError):
        disaggregate(ten_samples, resolution_exp_months=12)
    with pytest.raises(ValueError):
        disaggregate(ten_samples, resolution_exp_months=2)

    with pytest.raises(ValueError):
        disaggregate(
            ten_samples.select("paid_loss"),
            resolution_exp_months=1,
            resolution_dev_months=1,
            fields=["reported_loss"],
        )

    with pytest.raises(ValueError):
        disaggregate(
            ten_samples,
            resolution_exp_months=1,
            resolution_dev_months=1,
            period_weights=[0.5, 0.5],
        )
    with pytest.raises(ValueError):
        disaggregate(
            ten_samples,
            resolution_exp_months=1,
            resolution_dev_months=1,
            period_weights=[-1, 0, 1],
        )
    with pytest.raises(ValueError):
        disaggregate(
            ten_samples,
            resolution_exp_months=1,
            resolution_dev_months=1,
            period_weights=[1, 0, 1],
        )
    with pytest.raises(ValueError):
        disaggregate(
            ten_samples,
            resolution_exp_months=1,
            resolution_dev_months=1,
            period_weights={datetime.date(2020, 1, 1): [0, 0.5, 0.5]},
        )

    with pytest.raises(ValueError):
        disaggregate(
            ten_samples.derive_fields(paid_loss=lambda ob: np.exp(ob["paid_loss"])),
            resolution_exp_months=1,
            resolution_dev_months=1,
            interpolation_method="quadratic",
            extrapolate_first_period=False,
        )


def test_add_statics():
    base_tri_prem = add_statics(base_tri, premium_tri, ["earned_premium"])
    assert (
        base_tri_prem.cells[2].values["earned_premium"]
        == premium_tri.cells[0].values["earned_premium"]
    )


def test_cell_merge():
    merge_results = []
    obs = [
        Cell(
            datetime.date(2000, 1, 1),
            datetime.date(2000, 12, 31),
            datetime.date(2000, 12, 31),
            values={"earned_premium": 100},
            metadata=Metadata(details={"program_tag": "ABC"}),
        ),
        Cell(
            datetime.date(2000, 1, 1),
            datetime.date(2000, 12, 31),
            datetime.date(2000, 12, 31),
            values={"earned_premium": np.array([100, 200, 300])},
            metadata=Metadata(details={"program_tag": "XYZ"}),
        ),
    ]
    merge_results.append(summarize_cell_values(obs)["earned_premium"])
    assert (merge_results == np.array([200, 300, 400])).all()

    merge_inc_results = []
    inc_obs = [
        IncrementalCell(
            period_start=datetime.date(2000, 1, 1),
            period_end=datetime.date(2000, 12, 31),
            evaluation_date=datetime.date(2000, 12, 31),
            prev_evaluation_date=datetime.date(1999, 12, 31),
            values={"earned_premium": 100},
            metadata=Metadata(details={"program_tag": "ABC"}),
        ),
        IncrementalCell(
            period_start=datetime.date(2000, 1, 1),
            period_end=datetime.date(2000, 12, 31),
            evaluation_date=datetime.date(2000, 12, 31),
            prev_evaluation_date=datetime.date(1999, 12, 31),
            values={"earned_premium": np.array([100, 200, 300])},
            metadata=Metadata(details={"program_tag": "XYZ"}),
        ),
    ]
    merge_inc_results.append(summarize_cell_values(inc_obs)["earned_premium"])
    assert (merge_inc_results == np.array([200, 300, 400])).all()

    obs_diff_fields = obs + [
        Cell(
            datetime.date(2000, 1, 1),
            datetime.date(2000, 12, 31),
            datetime.date(2000, 12, 31),
            values={"loss_ratio": 0.6},
            metadata=Metadata(details={"program_tag": "ABC"}),
        )
    ]
    with pytest.raises(TriangleError):
        summarize_cell_values(obs_diff_fields)


def test_make_right_triangle():
    right_triangle = make_right_triangle(base_tri)
    extended_right_triangle = make_right_triangle(base_tri, dev_lags=range(3, 15, 3))
    right_inc_triangle = make_right_triangle(incremental_tri)
    extended_right_inc_triangle = make_right_triangle(
        incremental_tri, dev_lags=range(3, 15, 3)
    )

    assert len(right_triangle) == 3
    assert right_triangle.dev_lags("month") == [3.0, 6.0]
    assert not right_triangle.cells[0].values
    assert right_triangle.evaluation_dates == [
        date(2020, 12, 31),
        date(2021, 3, 31),
    ]
    assert len(extended_right_triangle) == 3 + 6
    assert extended_right_triangle.dev_lags("month") == [3.0, 6.0, 9.0, 12.0]

    assert len(right_inc_triangle) == 3
    assert right_inc_triangle.dev_lags("month") == [3.0, 6.0]
    assert not right_inc_triangle.cells[0].values
    assert right_inc_triangle.evaluation_dates == [
        date(2020, 12, 31),
        date(2021, 3, 31),
    ]
    assert sorted({cell.prev_evaluation_date for cell in right_inc_triangle}) == [
        datetime.date(2020, 9, 30),
        datetime.date(2020, 12, 31),
    ]
    assert len(extended_right_inc_triangle) == 3 + 6
    assert extended_right_inc_triangle.dev_lags("month") == [3.0, 6.0, 9.0, 12.0]


def test_make_right_diagonal():
    eval_dates = [datetime.date(2020, 12, 31), datetime.date(2021, 3, 31)]
    prev_eval_dates = [datetime.date(2020, 9, 30), datetime.date(2020, 12, 31)]

    diagonal = make_right_diagonal(base_tri, eval_dates)
    diagonal_inc = make_right_diagonal(incremental_tri, eval_dates)

    assert len(diagonal) == 6
    assert diagonal.dev_lags(unit="month") == [3.0, 6.0, 9.0, 12.0]
    assert diagonal.evaluation_dates == eval_dates

    assert len(diagonal_inc) == 6
    assert diagonal_inc.dev_lags(unit="month") == [3.0, 6.0, 9.0, 12.0]
    assert diagonal_inc.evaluation_dates == eval_dates
    assert (
        sorted({cell.prev_evaluation_date for cell in diagonal_inc}) == prev_eval_dates
    )


def test_add_dev_lag():
    test_cell = base_tri.cells[0]
    assert extend._add_dev_lag(test_cell, 28, "day") == datetime.date(2020, 4, 28)
    assert extend._add_dev_lag(test_cell, 28, "timedelta") == datetime.date(2020, 4, 28)

    with pytest.raises(ValueError):
        extend._add_dev_lag(test_cell, 2, "quarter")


def test_incremental_transform():
    assert incremental_tri == base_tri.to_incremental()
    assert base_tri == incremental_tri.to_cumulative()
    # triangle is already incremental
    incremental_tri.to_incremental() == incremental_tri

    incremental_tri_shifted_period_start = incremental_tri.replace(
        period_start=lambda ob: date_utils.resolution_delta(
            ob.period_start, (2, "day"), negative=True
        )
    )
    # check if prev_evaluation_date = period_start - 1 day for the first cell
    with pytest.raises(TriangleError):
        to_cumulative(incremental_tri_shifted_period_start)

    incremental_tri_shifted_eval_date = incremental_tri.replace(
        evaluation_date=lambda ob: date_utils.resolution_delta(
            ob.evaluation_date, (1, "day")
        )
    )
    # check if evaluation_date = prev_evaluation_date from previous cell
    with pytest.raises(TriangleError):
        to_cumulative(incremental_tri_shifted_eval_date)

    base_tri_different_fields = Triangle(
        [base_tri.select("paid_loss").cells[0]]
        + base_tri.derive_fields(earned_premium=lambda ob: 1e3).cells[1:]
    )
    incremental_tri_different_fields = Triangle(
        [incremental_tri.select("paid_loss").cells[0]]
        + incremental_tri.derive_fields(earned_premium=lambda ob: 1e3).cells[1:]
    )
    # different value keys for consecutive cells
    with pytest.raises(TriangleError):
        to_incremental(base_tri_different_fields)
    with pytest.raises(TriangleError):
        to_cumulative(incremental_tri_different_fields)


def test_aggregation():
    quad_tri = Triangle(
        [
            cell.replace(metadata=Metadata(details={"coverage": cov, "state": state}))
            for cell, cov, state in product(base_tri.cells, ["BI", "PD"], ["NY", "FL"])
        ]
    )

    double_tri = Triangle(
        [
            cell.replace(
                metadata=Metadata(details={"state": state}),
                values={k: 2 * v for k, v in cell.values.items()},
            )
            for state, cell in product(["NY", "FL"], base_tri.cells)
        ]
    )

    state_split = split(quad_tri, ["state"])
    assert ("NY",) in state_split
    assert ("FL",) in state_split

    state_agg = tlz.valmap(summarize, state_split)
    state_combined = Triangle([cell for val in state_agg.values() for cell in val])
    assert state_combined == double_tri


def test_make_pred_triangle():
    init_triangle = Triangle(
        [
            Cell(
                period_start=datetime.date(2000, 1, 1),
                period_end=datetime.date(2000, 12, 31),
                evaluation_date=datetime.date(2000, 12, 31),
                values={"earned_premium": 1e6},
            ),
            Cell(
                period_start=datetime.date(2000, 1, 1),
                period_end=datetime.date(2000, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                values={"earned_premium": 1e6},
            ),
            Cell(
                period_start=datetime.date(2001, 1, 1),
                period_end=datetime.date(2001, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                values={"earned_premium": 2e6},
            ),
            Cell(
                period_start=datetime.date(2001, 1, 1),
                period_end=datetime.date(2001, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                values={"earned_premium": 3e6},
                metadata=Metadata(risk_basis="Policy"),
            ),
        ]
    )
    init_inc_triangle = Triangle(
        [
            IncrementalCell(
                period_start=datetime.date(2000, 1, 1),
                period_end=datetime.date(2000, 12, 31),
                evaluation_date=datetime.date(2000, 12, 31),
                prev_evaluation_date=datetime.date(1999, 12, 31),
                values={"earned_premium": 1e6},
            ),
            IncrementalCell(
                period_start=datetime.date(2000, 1, 1),
                period_end=datetime.date(2000, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                prev_evaluation_date=datetime.date(2000, 12, 31),
                values={"earned_premium": 1e6},
            ),
            IncrementalCell(
                period_start=datetime.date(2001, 1, 1),
                period_end=datetime.date(2001, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                prev_evaluation_date=datetime.date(2000, 12, 31),
                values={"earned_premium": 2e6},
            ),
            IncrementalCell(
                period_start=datetime.date(2001, 1, 1),
                period_end=datetime.date(2001, 12, 31),
                evaluation_date=datetime.date(2001, 12, 31),
                prev_evaluation_date=datetime.date(2000, 12, 31),
                values={"earned_premium": 3e6},
                metadata=Metadata(risk_basis="Policy"),
            ),
        ]
    )
    pred_tri = make_pred_triangle_with_init(
        init_triangle=init_triangle,
        pred_triangle=None,
        max_dev_lag=(4, "years"),
        eval_resolution=(1, "year"),
        max_eval_date=datetime.date(2004, 12, 31),
    )
    pred_inc_tri = make_pred_triangle_with_init(
        init_triangle=init_inc_triangle,
        pred_triangle=None,
        max_dev_lag=(4, "years"),
        eval_resolution=(1, "year"),
        max_eval_date=datetime.date(2004, 12, 31),
    )
    pred_tri_complement = make_pred_triangle_complement(
        init_triangle=init_triangle,
        static_fields=["earned_premium"],
        max_dev_lag=60,
    )
    pred_inc_tri_complement = make_pred_triangle_complement(
        init_triangle=init_inc_triangle,
        static_fields=["earned_premium"],
        max_dev_lag=60,
    )

    assert len(pred_tri) == 9
    assert min(pred_tri.evaluation_dates) == datetime.date(2002, 12, 31)
    assert max(pred_tri.evaluation_dates) == datetime.date(2004, 12, 31)
    assert max(pred_tri.dev_lags()) == 48
    assert min(pred_tri.dev_lags()) == 12

    assert len(pred_inc_tri) == 9
    assert min(pred_inc_tri.evaluation_dates) == datetime.date(2002, 12, 31)
    assert max(pred_inc_tri.evaluation_dates) == datetime.date(2004, 12, 31)
    assert max(pred_inc_tri.dev_lags()) == 48
    assert min(pred_inc_tri.dev_lags()) == 12
    assert sorted({cell.prev_evaluation_date for cell in pred_inc_tri}) == [
        datetime.date(2001, 12, 31),
        datetime.date(2002, 12, 31),
        datetime.date(2003, 12, 31),
    ]

    assert len(pred_tri_complement) == 18
    assert min(pred_tri_complement.evaluation_dates) == datetime.date(2002, 12, 31)
    assert max(pred_tri_complement.evaluation_dates) == datetime.date(2006, 12, 31)
    assert max(pred_tri_complement.dev_lags()) == 60
    assert min(pred_tri_complement.dev_lags()) == 12
    assert pred_tri_complement.cells[0].values["earned_premium"] == 1e6

    assert len(pred_inc_tri_complement) == 18
    assert min(pred_inc_tri_complement.evaluation_dates) == datetime.date(2002, 12, 31)
    assert max(pred_inc_tri_complement.evaluation_dates) == datetime.date(2006, 12, 31)
    assert max(pred_inc_tri_complement.dev_lags()) == 60
    assert min(pred_inc_tri_complement.dev_lags()) == 12
    assert sorted({cell.prev_evaluation_date for cell in pred_inc_tri_complement}) == [
        datetime.date(year, 12, 31) for year in range(2001, 2006)
    ]
    assert pred_inc_tri_complement.cells[0].values["earned_premium"] == 1e6

    # Must specify either `max_eval` or `max_dev_lag`
    with pytest.raises(Exception):
        make_pred_triangle(
            metadata_sets=init_triangle.metadata,
            min_period=init_triangle.periods[0][0],
            max_period=init_triangle.periods[-1][-1],
            exp_resolution=(period_resolution(init_triangle), "months"),
            eval_resolution=(eval_date_resolution(init_triangle), "months"),
            exp_origin=init_triangle.periods[0][0] - datetime.timedelta(days=1),
            eval_origin=init_triangle.evaluation_dates[0] - datetime.timedelta(days=1),
            min_dev_lag=(init_triangle.dev_lags()[0], "months"),
            min_eval=init_triangle.evaluation_dates[0],
        )
    # Check for different evaluation dates
    with pytest.raises(TriangleError):
        make_pred_triangle_complement(
            init_triangle.replace(evaluation_date=datetime.date(2022, 12, 31))
        )
    # When `pred_triangle` is specified, other input variables shouldn't be provided
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle,
            pred_triangle=pred_tri,
            max_eval_date=datetime.date(2022, 12, 31),
        )
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle,
            pred_triangle=pred_tri,
            max_dev_lag=(5, "years"),
        )
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle,
            pred_triangle=pred_tri,
            eval_resolution=(1, "year"),
        )
    # When `pred_triangle` is not specified, other input variables should be provided
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(init_triangle=init_triangle)
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle, max_dev_lag=(5, "year")
        )
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle,
            max_dev_lag=(100, "week"),
            eval_resolution=(1, "year"),
        )
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_triangle,
            max_dev_lag=(5, "year"),
            eval_resolution=(4, "week"),
        )
    # Check cell type consistency
    with pytest.raises(TriangleError):
        make_pred_triangle_with_init(
            init_triangle=init_inc_triangle,
            pred_triangle=pred_tri,
        )


def test_make_pred_triangle_complement():
    """The complement must be able to handle tricky triangle shapes. This comes
    in handy when we have multiple development models for different dev lags."""
    with open("test/test_data/meyers_triangle.json", "r") as infile:
        paper_tri = json_to_triangle(infile)
    upper_left = paper_tri.clip(max_eval=date(1997, 12, 31))
    extra_cells = paper_tri.clip(
        min_period=date(1996, 1, 1), min_eval=date(1998, 12, 31), max_dev=24
    )
    oddly_shaped_triangle = upper_left + extra_cells
    pred_target = make_pred_triangle_complement(oddly_shaped_triangle)
    assert (
        date(1998, 12, 31) in pred_target.evaluation_dates
    )  # these are needed in earlier periods
    assert (
        date(1999, 12, 31) in pred_target.evaluation_dates
    )  # these are needed in earlier periods
    # be sure we get the whole rectangle
    assert len(oddly_shaped_triangle) + len(pred_target) == len(
        oddly_shaped_triangle.periods
    ) * len(oddly_shaped_triangle.dev_lags())


def test_make_pred_triangle_complement_holes():
    """When $0 losses are filtered out from dev lag 0, we should know not to
    include that dev lag 0 cell in the complement"""
    program_tri = binary_to_triangle("test/test_data/holey_init_tri.trib")
    complement = make_pred_triangle_complement(program_tri)
    assert 0 not in complement.dev_lags()


def test_make_pred_triangle_complement_eval_date_override():
    """Make sure that make_pred_triangle_complement still generates correct
    triangles when user inputs correct manual override."""
    test_triangle = Triangle(
        [
            CumulativeCell(
                period_start=date(2023, 1, 1),
                period_end=date(2023, 3, 31),
                evaluation_date=date(2023, 3, 31),
                values={
                    "earned_premium": 5,
                    "paid_loss": 1000,
                    "reported_loss": 10000,
                },
            )
        ]
    )
    complement = make_pred_triangle_complement(
        test_triangle, max_dev_lag=18, eval_date_resolution_override=3
    )

    assert len(complement) == 6
    assert complement.evaluation_dates == [
        datetime.date(2023, 6, 30),
        datetime.date(2023, 9, 30),
        datetime.date(2023, 12, 31),
        datetime.date(2024, 3, 31),
        datetime.date(2024, 6, 30),
        datetime.date(2024, 9, 30),
    ]
