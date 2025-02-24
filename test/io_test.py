import datetime
import math
import tempfile
import warnings
from typing import get_args

import awswrangler as wr
import boto3
import chainladder as cl
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

from bermuda import Cell, Metadata, Triangle
from bermuda.base.cell import CellValue
from bermuda.io import (
    array_data_frame_to_triangle,
    array_triangle_builder,
    binary_to_triangle,
    chain_ladder_to_triangle,
    json_string_to_triangle,
    json_to_triangle,
    long_csv_to_triangle,
    statics_data_frame_to_triangle,
    triangle_to_array_data_frame,
    triangle_to_binary,
    triangle_to_chain_ladder,
    triangle_to_json,
    triangle_to_long_csv,
    triangle_to_long_data_frame,
    triangle_to_right_edge_data_frame,
    triangle_to_wide_csv,
    triangle_to_wide_data_frame,
    wide_csv_to_triangle,
    wide_data_frame_to_triangle,
)
from bermuda.io.data_frame_input import INDEX_CUM_COLUMNS, _check_index_columns
from bermuda.io.data_frame_output import _common_field_length

from .triangle_test import base_tri, incremental_tri

incremental_tri_slices = incremental_tri + incremental_tri.derive_metadata(
    risk_basis="Policy"
)

json_string_cum = (
    '{"slices": [{'
    '"cells": [{"period_start": "2020-01-01", "period_end": "2020-03-31", '
    '"evaluation_date": "2020-03-31", '
    '"values": {"paid_loss": 100, "reported_loss": 200}}, '
    '{"period_start": "2020-01-01", "period_end": "2020-03-31", '
    '"evaluation_date": "2020-06-30", '
    '"values": {"paid_loss": 60, "reported_loss": 70}}]}]}'
)
json_string_inc = (
    '{"slices": [{'
    '"cells": [{"period_start": "2020-01-01", "period_end": "2020-03-31", '
    '"evaluation_date": "2020-03-31", "prev_evaluation_date": "2019-12-31", '
    '"values": {"paid_loss": 100, "reported_loss": 200}}, '
    '{"period_start": "2020-01-01", "period_end": "2020-03-31", '
    '"evaluation_date": "2020-06-30", "prev_evaluation_date": "2020-03-31", '
    '"values": {"paid_loss": 60, "reported_loss": 70}}]}]}'
)


@pytest.fixture
def tri_np_values():
    return Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2021, 3, 31),
                values={
                    "paid_loss": np.array([10_000, 20_000]),
                    "reported_loss": np.array([20_000, 30_000]),
                    "earned_premium": 50e3,
                },
            )
        ]
    )


def test_json_loads():
    cum_triangle = json_string_to_triangle(json_string_cum)
    inc_triangle = json_string_to_triangle(json_string_inc)

    assert cum_triangle.cells[0].__class__.__name__ == "CumulativeCell"
    assert inc_triangle.cells[0].__class__.__name__ == "IncrementalCell"
    assert cum_triangle.periods == [
        (datetime.date(2020, 1, 1), datetime.date(2020, 3, 31))
    ]
    assert inc_triangle.periods == [
        (datetime.date(2020, 1, 1), datetime.date(2020, 3, 31))
    ]
    assert cum_triangle.dev_lags() == [0.0, 3.0]
    assert inc_triangle.dev_lags() == [0.0, 3.0]


def test_json_load():
    parsed_tri = json_to_triangle("test/test_data/set_triangle_serde.json")
    assert parsed_tri == base_tri

    raw_tri = json_to_triangle("test/test_data/slice_triangle_serde.json")
    assert raw_tri == base_tri

    print("HMMM")
    inc_tri = json_to_triangle("test/test_data/incremental_triangle.json")
    print("HMMM")
    for cell in inc_tri:
        print(cell)
    print("===================================")
    for cell in incremental_tri_slices:
        print(cell)
    assert inc_tri == incremental_tri_slices


def test_wide_csv_load():
    meyers_tri = wide_csv_to_triangle(
        "test/test_data/meyers_wide.csv",
        field_cols=["earned_premium", "reported_loss", "paid_loss"],
    )
    assert len(meyers_tri) == 100
    assert len(meyers_tri.slices) == 1
    assert meyers_tri.fields == ["earned_premium", "paid_loss", "reported_loss"]

    inc_tri = wide_csv_to_triangle(
        "test/test_data/incremental_wide.csv",
        field_cols=["reported_loss", "paid_loss"],
    )
    assert len(inc_tri) == 12
    assert len(inc_tri.slices) == 2
    assert inc_tri.fields == ["paid_loss", "reported_loss"]

    csv_path = "test/test_data/meyers_wide.csv"
    with pytest.raises(Exception):
        wide_csv_to_triangle(csv_path, field_cols=None, detail_cols=None)
    with pytest.raises(Exception):
        wide_csv_to_triangle(csv_path, field_cols=["state"], detail_cols=["state"])
    with pytest.raises(Exception):
        wide_csv_to_triangle(
            csv_path, detail_cols=["state"], loss_detail_cols=["state", "class_code"]
        )


def test_long_csv_load():
    meyers_tri = long_csv_to_triangle("test/test_data/meyers_long.csv")
    assert len(meyers_tri) == 100
    assert len(meyers_tri.slices) == 1
    assert meyers_tri.fields == ["earned_premium", "paid_loss", "reported_loss"]

    inc_tri = long_csv_to_triangle("test/test_data/incremental_long.csv")
    assert len(inc_tri) == 12
    assert len(inc_tri.slices) == 2
    assert inc_tri.fields == ["paid_loss", "reported_loss"]

    # Test when columns are missing or bad data types
    # FIXME: I don't like the idea of creating a bunch of these modified CSVs
    #        for each exception condition. However, I haven't yet been able to
    #        make this work with tempfiles since `long_csv_to_triangle` reads
    #        the CSV twice (causes issues with in-memory files). If there's a
    #        way to do this more easily, then we can rememve these CSVs.
    df_base_incr = pd.read_csv("test/test_data/incremental_long.csv")
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        # test when there's no `field` column
        df_base_incr.drop(["field"], axis=1).to_csv(f.name, index=False)
        with pytest.raises(Exception, match="must have a column `field`"):
            long_csv_to_triangle(f.name)

        # test when `field` column isn't a string
        df_non_string_field = df_base_incr.copy()
        df_non_string_field["field"] = 1
        df_non_string_field.to_csv(f.name, index=False)
        with pytest.raises(Exception, match="field column must be a string"):
            long_csv_to_triangle(f.name)

        # test when there's no `value` column
        df_base_incr.drop(["value"], axis=1).to_csv(f.name, index=False)
        with pytest.raises(Exception, match="must have a column `value`"):
            long_csv_to_triangle(f.name)

        # test when `value` column is non-numeric
        df_non_numeric_value = df_base_incr.copy()
        df_non_numeric_value["value"] = "some string"
        df_non_numeric_value.to_csv(f.name, index=False)
        with pytest.raises(Exception, match="value column must be numeric"):
            long_csv_to_triangle(f.name)

        # test when `field` value is repeated (incremental triangle)
        pd.DataFrame(
            {
                "period_start": ["2020-01-01", "2020-01-01", "2020-01-01"],
                "period_end": ["2020-03-31", "2020-03-31", "2020-03-31"],
                "evaluation_date": ["2020-03-31", "2020-03-31", "2020-03-31"],
                "prev_evaluation_date": ["2019-12-31", "2019-12-31", "2019-12-31"],
                "risk_basis": ["Policy", "Policy", "Policy"],
                "field": ["paid_loss", "reported_loss", "paid_loss"],
                "value": [100.0, 200.0, 200.0],
            }
        ).to_csv(f.name, index=False)
        with pytest.raises(Exception, match=r"Field .* is already present"):
            long_csv_to_triangle(f.name)

        # test when `field` is repeated (meyers triangle)
        pd.DataFrame(
            {
                "period_start": ["2020-01-01", "2020-01-01", "2020-01-01"],
                "period_end": ["2020-03-31", "2020-03-31", "2020-03-31"],
                "evaluation_date": ["2020-03-31", "2020-03-31", "2020-03-31"],
                # "prev_evaluation_date": ["2019-12-31", "2019-12-31", "2019-12-31"],
                "risk_basis": ["Policy", "Policy", "Policy"],
                "field": ["paid_loss", "reported_loss", "paid_loss"],
                "value": [100.0, 200.0, 200.0],
            }
        ).to_csv(f.name, index=False)
        with pytest.raises(Exception, match=r"Field .* is already present"):
            long_csv_to_triangle(f.name)


def test_wide_data_frame_write():
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    df = triangle_to_wide_data_frame(meyers_tri)
    assert df.shape == (100, 11)

    inc_tri = json_to_triangle("test/test_data/incremental_triangle.json")
    df = triangle_to_wide_data_frame(inc_tri)
    assert df.shape == (12, 7)


def test_long_data_frame_write():
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    df = triangle_to_long_data_frame(meyers_tri)
    assert df.shape == (300, 10)

    inc_tri = json_to_triangle("test/test_data/incremental_triangle.json")
    df = triangle_to_long_data_frame(inc_tri)
    assert df.shape == (24, 7)


def test_triangle_to_json():
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    json_str = triangle_to_json(meyers_tri)
    assert isinstance(json_str, str)

    inc_tri = json_to_triangle("test/test_data/incremental_triangle.json")
    json_str_inc = triangle_to_json(inc_tri)
    assert isinstance(json_str_inc, str)


def test_triangle_json_df_triangle_conversion():
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    df = meyers_tri.to_data_frame()
    df2 = triangle_to_wide_data_frame(meyers_tri)
    assert wide_data_frame_to_triangle(
        df, field_cols=["earned_premium", "reported_loss", "paid_loss"]
    )
    assert wide_data_frame_to_triangle(
        df2, field_cols=["earned_premium", "reported_loss", "paid_loss"]
    )


def test_missing_value_load():
    meyers_wide_df = pd.read_csv(
        "test/test_data/meyers_wide.csv", parse_dates=INDEX_CUM_COLUMNS
    )
    meyers_wide_df.loc[0, "earned_premium"] = np.NaN
    meyers_tri_from_wide = wide_data_frame_to_triangle(
        meyers_wide_df, field_cols=["earned_premium", "reported_loss", "paid_loss"]
    )
    assert "earned_premium" not in meyers_tri_from_wide.cells[0]

    meyers_long_df = pd.read_csv(
        "test/test_data/meyers_long.csv", parse_dates=INDEX_CUM_COLUMNS
    )

    with pytest.raises(Exception):
        _check_index_columns(meyers_long_df.drop(columns=["evaluation_date"]))

    with pytest.raises(Exception):
        _check_index_columns(meyers_long_df.astype({"evaluation_date": str}))

    meyers_long_df["prev_evaluation_date"] = len(meyers_long_df.index) * ["date"]
    with pytest.raises(Exception):
        _check_index_columns(meyers_long_df)


def test_data_frame_output_utils():
    test_cell = Cell(
        period_start=datetime.date(2000, 1, 1),
        period_end=datetime.date(2000, 3, 31),
        evaluation_date=datetime.date(2000, 3, 31),
        values={
            "paid_loss": np.array([[100, 200, 150], [110, 210, 200]]),
            "reported_loss": np.array([300, 500]),
            "earned_premium": np.array([1000, 1000, 1200]),
        },
    )
    with pytest.raises(ValueError):
        _common_field_length(test_cell, field_names=["paid_loss"])

    with pytest.raises(ValueError):
        _common_field_length(test_cell, field_names=["reported_loss", "earned_premium"])


@mock_aws
def test_triangle_to_s3_csv_no_bucket():
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="test_bucket")
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    triangle_to_wide_csv(meyers_tri, filename="s3://test_bucket/test_bermuda_tri.csv")
    wide_result_tri = wide_csv_to_triangle(
        "s3://test_bucket/test_bermuda_tri.csv",
        field_cols=["earned_premium", "reported_loss", "paid_loss"],
    )
    triangle_to_long_csv(
        meyers_tri, filename="s3://test_bucket/test_bermuda_tri_long.csv"
    )
    long_result_tri = long_csv_to_triangle("s3://test_bucket/test_bermuda_tri_long.csv")
    assert meyers_tri == wide_result_tri == long_result_tri


def test_data_frame_input_metadata_details():
    metadata = Metadata(
        details={
            "my detail col": "default detail val",
        },
        loss_details={
            "my loss detail col": "default loss_detail val",
        },
    )
    res = wide_csv_to_triangle(
        "test/test_data/meyers_wide_w_details.csv",
        detail_cols=["detail_col_1", "loss_detail_col_1"],
        loss_detail_cols=["loss_detail_col_1"],
        metadata=metadata,
    )

    assert all(
        [
            (c.details["my detail col"] == "default detail val")
            and (c.loss_details["my loss detail col"] == "default loss_detail val")
            for c in res.cells
        ]
    )


def test_wide_data_frame_to_triangle_missing_details():
    df = pd.DataFrame(
        {
            "period_start": [np.datetime64(datetime.date(2023, 1, 1))] * 2,
            "period_end": [np.datetime64(datetime.date(2023, 1, 31))] * 2,
            "evaluation_date": [np.datetime64(datetime.date(2023, 1, 31))] * 2,
            "risk_basis": "Accident",
            "currency": "USD",
            "country": "US",
            "reported_loss": [1] * 2,
            "detail": [True, None],
        }
    )

    # Ensure no warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        triangle = wide_data_frame_to_triangle(df, field_cols=["reported_loss"])

    assert len(triangle.slices) == 2


def test_scenario_triangle_to_long_csv(tri_np_values):
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_long_csv(tri_np_values, f.name)
        loaded_tri = long_csv_to_triangle(f.name)
    assert loaded_tri == tri_np_values


def test_single_triangle_to_long_csv():
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_long_csv(base_tri, f.name)
        loaded_tri = long_csv_to_triangle(f.name)
    assert loaded_tri == base_tri


def test_scenario_triangle_to_wide_csv(tri_np_values):
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_wide_csv(tri_np_values, f.name)
        loaded_tri = wide_csv_to_triangle(
            f.name,
            field_cols=["paid_loss", "reported_loss", "earned_premium"],
            collapse_fields=["earned_premium"],
        )
    assert loaded_tri == tri_np_values


def test_single_triangle_to_wide_csv():
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_wide_csv(base_tri, f.name)
        loaded_tri = wide_csv_to_triangle(
            f.name, field_cols=["paid_loss", "reported_loss"]
        )
    assert loaded_tri == base_tri


def test_incremental_to_long_csv():
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_long_csv(incremental_tri, f.name)
        loaded_tri = long_csv_to_triangle(f.name)
    assert loaded_tri == incremental_tri


def test_incremental_to_wide_csv():
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_wide_csv(incremental_tri, f.name)
        loaded_tri = wide_csv_to_triangle(
            f.name, field_cols=["paid_loss", "reported_loss"]
        )
    assert loaded_tri == incremental_tri


def test_triangle_to_long_data_frame_clean_field_dicts_np_values(tri_np_values):
    res = triangle_to_long_data_frame(tri_np_values)
    # tri_np_values has 2 values items per value key (e.g. paid_loss), so
    # we expect the number of scenarios to be 2 per value metric
    # drop earned_premium where it's a single value and scenario should be NA
    assert (
        res.dropna().groupby("field").apply(lambda s: s.scenario.nunique() == 2).all()
    )


def test_triangle_to_binary():
    with tempfile.NamedTemporaryFile(suffix=".trib") as f:
        triangle_to_binary(base_tri, f.name)
        round_trip_base_tri = binary_to_triangle(f.name)
        assert round_trip_base_tri == base_tri


def test_triangle_to_compressed_binary():
    with tempfile.NamedTemporaryFile(suffix=".tribc") as f:
        triangle_to_binary(base_tri, f.name, compress=True)
        round_trip_tri = binary_to_triangle(f.name)
        assert round_trip_tri == base_tri


def test_triangle_to_binary_array(tri_np_values):
    with tempfile.NamedTemporaryFile(suffix=".trib") as f:
        triangle_to_binary(tri_np_values, f.name)
        round_trip_tri = binary_to_triangle(f.name)
        assert round_trip_tri == tri_np_values


def test_triangle_to_binary_incremental():
    with tempfile.NamedTemporaryFile(suffix=".trib") as f:
        triangle_to_binary(incremental_tri_slices, f.name)
        round_trip_tri = binary_to_triangle(f.name)
        assert round_trip_tri == incremental_tri_slices


def test_binary_magic_check():
    with pytest.raises(Exception):
        binary_to_triangle("test/test_data/incremental_long.csv")


def test_triangle_to_binary_exotic_metadata():
    weird_meta_tri = base_tri.derive_metadata(
        is_a_good_program=True,
        program_inception_date=datetime.date(2013, 1, 1),
        program_is_round=math.pi,
        program_message="Special Message δεψβ",
        mystery_value=None,
    )
    with tempfile.NamedTemporaryFile(suffix=".trib") as f:
        triangle_to_binary(weird_meta_tri, f.name)
        round_trip_tri = binary_to_triangle(f.name)
        assert round_trip_tri == weird_meta_tri


@mock_aws
def test_s3_round_trip():
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="test_bucket")
    triangle_to_binary(base_tri, "s3://test_bucket/test_base.trib")
    round_trip_base_tri = binary_to_triangle("s3://test_bucket/test_base.trib")
    assert round_trip_base_tri == base_tri


def test_mixed_field_tri_to_csv():
    common_values = {"earned_premium": 1000, "reported_loss": 100}
    irregular_tri = Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2020, 12, 31),
                values=common_values,
            ),
            Cell(
                period_start=datetime.date(2021, 1, 1),
                period_end=datetime.date(2021, 12, 31),
                evaluation_date=datetime.date(2021, 12, 31),
                values={**common_values, "extra_field": 100},
            ),
        ]
    )
    with tempfile.NamedTemporaryFile("w") as f:
        triangle_to_wide_csv(irregular_tri, f.name)
        loaded_tri = wide_csv_to_triangle(f.name, detail_cols=[])
    assert loaded_tri == irregular_tri


def test_trib_triangle_value_io():
    cell_types = get_args(CellValue)
    for cell_type in cell_types:
        tri = Triangle(
            [
                Cell(
                    period_start=datetime.date(2020, 1, 1),
                    period_end=datetime.date(2020, 12, 31),
                    evaluation_date=datetime.date(2020, 12, 31),
                    values={
                        "paid_loss": cell_type(100)
                        if cell_type is not type(None)
                        else None
                    },
                )
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".trib") as f:
            triangle_to_binary(tri, f.name)
            round_trip_tri = binary_to_triangle(f.name)
            assert round_trip_tri == tri


def test_from_chain_ladder():
    chain_ladder_tri = cl.load_sample("clrd")
    bermuda_tri = chain_ladder_to_triangle(chain_ladder_tri)
    array_tri = cl.load_sample("raa")
    bermuda_tri_array = chain_ladder_to_triangle(array_tri)
    assert isinstance(bermuda_tri, Triangle)
    assert bermuda_tri.fields == [
        "BulkLoss",
        "CumPaidLoss",
        "EarnedPremCeded",
        "EarnedPremDIR",
        "EarnedPremNet",
        "IncurLoss",
    ]
    assert isinstance(bermuda_tri_array, Triangle)
    assert len(bermuda_tri_array) == 55


def test_to_chain_ladder():
    raw_tri = json_to_triangle("test/test_data/slice_triangle_serde.json")
    cl_tri = triangle_to_chain_ladder(raw_tri)
    assert isinstance(cl_tri, cl.Triangle)


def test_multiple_triangle_chain_ladder(tri_np_values):
    meyers_tri = json_to_triangle("test/test_data/meyers_triangle.json")
    slice_tri = json_to_triangle("test/test_data/slice_triangle_serde.json")
    incremental_tri = json_to_triangle("test/test_data/incremental_triangle.json")
    holey_tri = binary_to_triangle("test/test_data/holey_init_tri.trib")
    tri_list = [meyers_tri, slice_tri, incremental_tri, holey_tri, tri_np_values]
    for triangle in tri_list:
        cl_tri = triangle_to_chain_ladder(triangle)
        assert isinstance(cl_tri, cl.Triangle)


def test_static_df_to_triangle():
    statics_df = pd.DataFrame(
        [
            {"period": period[0], "earned_premium": 100 * ee, "earned_exposure": ee}
            for period, ee in zip(base_tri.periods, range(1, 7))
        ]
    )
    statics_tri = statics_data_frame_to_triangle(statics_df)
    assert isinstance(statics_tri, Triangle)
    assert len(statics_tri) == statics_df.shape[0]
    assert set(statics_tri.fields) == {"earned_premium", "earned_exposure"}
    assert statics_tri.evaluation_date == max(statics_tri.periods)[-1]


def test_array_df_to_triangle():
    annual_df = pd.read_csv("test/test_data/annual_array_tri.csv")
    quarterly_df = pd.read_csv("test/test_data/quarterly_array_tri.csv")
    quarterly_no_header_df = pd.read_csv(
        "test/test_data/quarterly_array_tri_no_header.csv"
    )
    quarterly_bermuda_dev_convention_df = pd.read_csv(
        "test/test_data/quarterly_array_tri_bermuda_dev_convention.csv"
    )
    monthly_df = pd.read_csv("test/test_data/monthly_array_tri.csv")

    annual_tri = array_data_frame_to_triangle(
        annual_df, "paid_loss", dev_lag_from_period_end=False
    )
    quarterly_tri = array_data_frame_to_triangle(
        quarterly_df, "reported_loss", dev_lag_from_period_end=False
    )
    quarterly_no_header_tri = array_data_frame_to_triangle(
        quarterly_no_header_df, "reported_loss", dev_lag_from_period_end=False
    )
    quarterly_bermuda_dev_convention_tri = array_data_frame_to_triangle(
        quarterly_bermuda_dev_convention_df,
        "reported_loss",
        dev_lag_from_period_end=True,
    )
    monthly_tri = array_data_frame_to_triangle(
        monthly_df, "reported_loss", dev_lag_from_period_end=False
    )

    multi_field = array_triangle_builder(
        [quarterly_df, quarterly_df],
        ["reported_loss", "paid_loss"],
        dev_lag_from_period_end=False,
    )

    assert isinstance(annual_tri, Triangle)
    assert isinstance(quarterly_tri, Triangle)
    assert isinstance(quarterly_no_header_tri, Triangle)
    assert isinstance(quarterly_bermuda_dev_convention_tri, Triangle)
    assert isinstance(monthly_tri, Triangle)

    assert len(annual_tri) == len(quarterly_tri) == len(monthly_tri) == 6
    assert annual_tri.periods[0] == (
        datetime.date(2018, 1, 1),
        datetime.date(2018, 12, 31),
    )
    assert quarterly_tri.periods[0] == (
        datetime.date(2018, 1, 1),
        datetime.date(2018, 3, 31),
    )
    assert monthly_tri.periods[0] == (
        datetime.date(2018, 1, 1),
        datetime.date(2018, 1, 31),
    )
    assert monthly_tri.dev_lags() == [0, 1, 2]
    assert quarterly_no_header_tri.dev_lags() == [0, 3, 6]
    assert quarterly_bermuda_dev_convention_tri.dev_lags() == [0, 3, 6]
    assert set(multi_field.fields) == {"reported_loss", "paid_loss"}
    assert len(multi_field) == 6


def test_triangle_to_array_df():
    paid_df = triangle_to_array_data_frame(base_tri, "paid_loss")
    reported_df = triangle_to_array_data_frame(base_tri, "reported_loss")

    right_edge_df = triangle_to_right_edge_data_frame(base_tri)

    assert isinstance(paid_df, pd.DataFrame)
    assert isinstance(reported_df, pd.DataFrame)
    assert isinstance(right_edge_df, pd.DataFrame)
    assert paid_df.shape == reported_df.shape == (3, 4)
    assert right_edge_df.shape == (3, 4)
