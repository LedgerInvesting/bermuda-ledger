from typing import Any

import awswrangler as wr
import numpy as np
import pandas as pd
import toolz as tlz

from ..base import Cell, IncrementalCell
from ..triangle import Triangle

__all__ = [
    "triangle_to_wide_csv",
    "triangle_to_long_csv",
    "triangle_to_wide_data_frame",
    "triangle_to_long_data_frame",
]


def triangle_to_wide_csv(tri: Triangle[Cell], filename: str) -> None:
    """Convert a Triangle to a wide-format CSV file.

    See `wide_data_frame_to_triangle` for details on the wide format.

    Args:
        tri: Triangle to convert.
        filename: Name of CSV file to create.
    """
    df = triangle_to_wide_data_frame(tri)
    # noinspection PyTypeChecker
    if filename.startswith("s3://"):
        wr.s3.to_csv(df, filename, index=False)
    else:
        df.to_csv(filename, index=False)


def triangle_to_long_csv(tri: Triangle[Cell], filename: str) -> None:
    """Convert a Triangle to a long-format CSV file.

    See `long_data_frame_to_triangle` for details on the long format.

    Args:
        tri: Triangle to convert.
        filename: Name of CSV file to create.
    """
    df = triangle_to_long_data_frame(tri)
    # noinspection PyTypeChecker
    if filename.startswith("s3://"):
        wr.s3.to_csv(df, filename, index=False)
    else:
        df.to_csv(filename, index=False)


def triangle_to_wide_data_frame(tri: Triangle[Cell]) -> pd.DataFrame:
    """Convert a Triangle to a wide-format Pandas `DataFrame`.

    See `wide_data_frame_to_triangle` for details on the wide format.

    Args:
        tri: Triangle to convert.
    """
    metadata_names = _all_metadata_names(tri)
    field_names = _all_fields(tri)
    rows = [
        row
        for cell in tri
        for row in _cell_to_wide_dict(cell, metadata_names, field_names)
    ]

    df = pd.DataFrame(rows)
    df = _drop_constant_scenario(df)
    df.period_start = pd.to_datetime(df.period_start)
    df.period_end = pd.to_datetime(df.period_end)
    df.evaluation_date = pd.PeriodIndex(
        _date_to_period_index(df.evaluation_date)
    ).to_timestamp()
    if tri.is_incremental:
        df.prev_evaluation_date = pd.PeriodIndex(
            _date_to_period_index(df.prev_evaluation_date)
        )
    return df


def triangle_to_long_data_frame(tri: Triangle[Cell]) -> pd.DataFrame:
    """Convert a Triangle to a long-format Pandas `DataFrame`.

    See `long_data_frame_to_triangle` for details on the long format.

    Args:
        tri: Triangle to convert.
    """
    metadata_names = _all_metadata_names(tri)
    rows = [row for cell in tri for row in _cell_to_long_dict(cell, metadata_names)]

    df = pd.DataFrame(rows)
    df = _drop_constant_scenario(df)
    df.period_start = pd.to_datetime(df.period_start)
    df.period_end = pd.to_datetime(df.period_end)
    df.evaluation_date = pd.PeriodIndex(_date_to_period_index(df.evaluation_date))
    if tri.is_incremental:
        df.prev_evaluation_date = pd.PeriodIndex(
            _date_to_period_index(df.prev_evaluation_date)
        )
    return df


def _cell_to_long_dict(cell: Cell, metadata_names: list[str]) -> list[dict[str, Any]]:
    metadata_raw_dict = cell.metadata.as_flat_dict()
    metadata_dict = {name: metadata_raw_dict.get(name) for name in metadata_names}
    base_dict = {
        "period_start": cell.period_start,
        "period_end": cell.period_end,
        "evaluation_date": cell.evaluation_date,
    }
    if isinstance(cell, IncrementalCell):
        base_dict.update({"prev_evaluation_date": cell.prev_evaluation_date})
    clean_field_dicts = _clean_field_dicts(cell)

    constant_fields = tlz.valfilter(
        lambda vals: isinstance(vals, (float, int)), cell.values
    )

    return [
        {
            **base_dict,
            **metadata_dict,
            "scenario": np.NaN if field in constant_fields else ndx + 1,
            "field": field,
            "value": float(value),
        }
        for ndx, field_dict in enumerate(clean_field_dicts)
        for field, value in field_dict.items()
    ]


def _cell_to_wide_dict(
    cell: Cell, metadata_names: list[str], field_names: list[str]
) -> list[dict[str, Any]]:
    metadata_raw_dict = cell.metadata.as_flat_dict()
    metadata_dict = {name: metadata_raw_dict.get(name) for name in metadata_names}
    clean_field_dicts = _clean_field_dicts(cell, field_names)
    base_dict = {
        "period_start": cell.period_start,
        "period_end": cell.period_end,
        "evaluation_date": cell.evaluation_date,
    }
    if isinstance(cell, IncrementalCell):
        base_dict.update({"prev_evaluation_date": cell.prev_evaluation_date})

    return [
        {
            **base_dict,
            "scenario": ndx + 1,
            **field_dict,
            **metadata_dict,
        }
        for ndx, field_dict in enumerate(clean_field_dicts)
    ]


def _clean_field_dicts(
    cell: Cell, field_names: list[str] = None
) -> list[dict[str, float]]:
    if field_names is None:
        field_names = list(cell.values.keys())
    common_length = _common_field_length(cell, field_names)
    field_dicts = []

    for ndx in range(common_length):
        field_dict = {}
        for field in field_names:
            value = cell.values.get(field)
            if isinstance(value, np.ndarray) and value.size > 1:
                field_dict[field] = value[ndx]
            elif value is not None:
                field_dict[field] = float(value)
        field_dicts.append(field_dict)

    return field_dicts


def _common_field_length(cell: Cell, field_names: list[str]) -> int:
    field_lengths = set()

    for field in field_names:
        value = cell.values.get(field)
        if isinstance(value, np.ndarray):
            if len(value.shape) > 1:
                raise ValueError("Multidimensional array values are not supported yet")
            field_lengths |= {value.size}
        else:
            field_lengths |= {1}

    # We don't want multiple non-1 field_lengths
    if field_lengths == {1}:
        return 1
    interesting_field_lengths = field_lengths - {1}
    if len(interesting_field_lengths) > 1:
        raise ValueError("Incompatible field lengths are not supported yet")
    return max(interesting_field_lengths)


def _all_fields(tri: Triangle[Cell]) -> list[str]:
    field_names = set()

    for cell in tri:
        cell_fields = set(cell.values.keys())
        field_names |= cell_fields

    return list(field_names)


def _all_metadata_names(tri: Triangle[Cell]) -> list[str]:
    metadata_col_names = set()
    all_metadata = tri.metadata

    for metadata in all_metadata:
        col_names = {k for k, v in metadata.as_flat_dict().items() if v is not None}
        metadata_col_names |= col_names

    return list(metadata_col_names)


def _date_to_period_index(dates: list) -> list[pd.Period]:
    return [pd.Period(year=d.year, month=d.month, day=d.day, freq="D") for d in dates]


def _drop_constant_scenario(df: pd.DataFrame) -> pd.DataFrame:
    scenario_values = df["scenario"].to_numpy()
    if np.all(scenario_values[0] == scenario_values) or np.all(
        pd.isnull(scenario_values)
    ):
        return df.drop(columns=["scenario"])
    else:
        return df
