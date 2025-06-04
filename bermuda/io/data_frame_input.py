import datetime
from typing import Any, Union
from warnings import warn

import awswrangler as wr
import numpy as np
import pandas as pd
import toolz as tlz

from ..base import CumulativeCell, IncrementalCell, Metadata
from ..triangle import Triangle

__all__ = [
    "wide_csv_to_triangle",
    "long_csv_to_triangle",
    "wide_data_frame_to_triangle",
    "long_data_frame_to_triangle",
]


INDEX_CUM_COLUMNS = ["period_start", "period_end", "evaluation_date"]
INDEX_COLUMNS = INDEX_CUM_COLUMNS + ["prev_evaluation_date"]

METADATA_COLUMNS = [
    "risk_basis",
    "country",
    "currency",
    "reinsurance_basis",
    "loss_definition",
    "per_occurrence_limit",
]

CORE_SET = set(INDEX_COLUMNS + METADATA_COLUMNS + ["scenario"])


def wide_csv_to_triangle(
    file_or_fname: str,
    field_cols: list[str] | None = None,
    detail_cols: list[str] | None = None,
    loss_detail_cols: list[str] | None = None,
    metadata: Metadata | None = None,
    collapse_fields: list[str] | None = None,
    **kwargs,
) -> Union[Triangle[CumulativeCell], Triangle[IncrementalCell]]:
    """
    Convert a wide-format CSV to a `Triangle`.

    For more details on wide-format CSVs/tables, see `wide_data_frame_to_triangle` (which
    this function is a thin wrapper for).

    Args:
        file_or_fname: Name of the CSV file or a handle to a file object.
        field_cols: Names of the columns that should be interpreted as fields.
        detail_cols: Names of the columns that should be interpreted as metadata details.
            At least one of `field_cols` or `detail_cols` must be supplied.
        loss_detail_cols: Names of the detail columns that should be interpreted as metadata
            loss details, rather than standard metadata details. By default, nothing is assumed
            to be a loss detail.
        metadata: Default metadata to apply to every cell in the `Triangle` (can be overridden
            by `detail_cols`).
        collapse_fields: A list of fields to collapse to a single value, rather than loading as
            an array with multiple values. This typically applies to `earned_premium` when multiple
            scenarios are being loaded.
        **kwargs: Extra arguments to `pd.read_csv`
    """
    # Read column names first
    if file_or_fname.startswith("s3://"):
        raw_df = wr.s3.read_csv(file_or_fname, nrows=1)
    else:
        raw_df = pd.read_csv(file_or_fname, nrows=1)
    parse_dates = (
        INDEX_COLUMNS if "prev_evaluation_date" in raw_df.columns else INDEX_CUM_COLUMNS
    )
    # Read full table with matching column names
    if file_or_fname.startswith("s3://"):
        df = wr.s3.read_csv(file_or_fname, parse_dates=parse_dates, **kwargs)
    else:
        df = pd.read_csv(file_or_fname, parse_dates=parse_dates, **kwargs)
    return wide_data_frame_to_triangle(
        df, field_cols, detail_cols, loss_detail_cols, metadata, collapse_fields
    )


def long_csv_to_triangle(
    file_or_fname: str, metadata: Metadata | None = None, **kwargs
) -> Union[Triangle[CumulativeCell], Triangle[IncrementalCell]]:
    """
    Convert a long-format CSV to a `Triangle`.

    For more details on long-format CSVs/tables, see `long_data_frame_to_triangle` (which
    this function is a thin wrapper for).

    Args:
        file_or_fname: Name of the CSV file or a handle to a file object.
        detail_cols: Names of the columns that should be interpreted as metadata details.
            At least one of `field_cols` or `detail_cols` must be supplied.
        loss_detail_cols: Names of the metadata columns that should be interpreted as metadata
            loss details, rather than standard metdata details. By default, nothing is assumed
            to be a loss detail.
        **kwargs: Extra arguments to `pd.read_csv`
    """
    # Read column names first
    if file_or_fname.startswith("s3://"):
        raw_df = wr.s3.read_csv(file_or_fname, nrows=1)
    else:
        raw_df = pd.read_csv(file_or_fname, nrows=1)
    parse_dates = (
        INDEX_COLUMNS if "prev_evaluation_date" in raw_df.columns else INDEX_CUM_COLUMNS
    )
    # Read full table with matching column names
    if file_or_fname.startswith("s3://"):
        df = wr.s3.read_csv(file_or_fname, parse_dates=parse_dates, **kwargs)
    else:
        df = pd.read_csv(file_or_fname, parse_dates=parse_dates, **kwargs)
    return long_data_frame_to_triangle(df, metadata=metadata)


def wide_data_frame_to_triangle(
    df: pd.DataFrame,
    field_cols: list[str] | None = None,
    detail_cols: list[str] | None = None,
    loss_detail_cols: list[str] | None = None,
    metadata: Metadata | None = None,
    collapse_fields: list[str] | None = None,
) -> Union[Triangle[CumulativeCell], Triangle[IncrementalCell]]:
    """
    Convert a wide-format `pd.DataFrame` to a `Triangle`.

    A wide-format `DataFrame` requires the following columns to be present: `period_start`,
    `period_end`, and `evaluation_date`. When the column 'prev_evaluation_date' is present,
    a triangle with incremental cells is generated. Otherwise, all triangle cells are cumulative.

    Each row of the `DataFrame` represents a single `Cell` of the `Triangle`.
    Any columns besides the three required columns are interpreted as either fields
    (i.e., `paid_loss`, `reported_loss`) or metadata (i.e., `per_occurrence_limit`,
    `currency`, `program_name`).

    This function doesn't make any assumptions about which columns are fields and which are
    metadata based on their names. Instead, at least one of `detail_cols` or `field_cols` must be
    supplied to the function. If only one of those arguments is supplied, all other columns are
    assumed to belong to the other argument.

    The `metadata` argument can be used to supply common metadata that may not be directly
    present in the `DataFrame`. For example, every `Cell` in the `Triangle` could be
    accident-basis, gross of reinsurance, including DCC. This could be supplied by a
    single `Metadata` object in the function call instead of adding several constant columns
    to the `DataFrame`.

    Args:
        df: Pandas `DataFrame` to convert to a `Triangle`.
        field_cols: Names of the columns that should be interpreted as fields.
        detail_cols: Names of the columns that should be interpreted as metadata details.
            At least one of `field_cols` or `detail_cols` must be supplied.
        loss_detail_cols: Names of the detail columns that should be interpreted as metadata
            loss details, rather than standard metadata details. By default, nothing is assumed
            to be a loss detail.
        metadata: Default metadata to apply to every cell in the `Triangle` (can be overridden
            by `detail_cols`).
        collapse_fields: A list of fields to collapse to a single value, rather than loading as
            an array with multiple values. This typically applies to `earned_premium` when multiple
            scenarios are being loaded.
    """
    _check_index_columns(df)
    if field_cols is None and detail_cols is None:
        raise Exception("At least one of field_cols or detail_cols must be supplied")
    if field_cols is not None and detail_cols is not None:
        if len(set(field_cols) & set(detail_cols)):
            raise Exception("field_cols and detail_cols must be disjoint")
    if field_cols is None:
        field_cols = list(set(df.columns.values) - CORE_SET - set(detail_cols))
    if detail_cols is None:
        detail_cols = list(set(df.columns.values) - CORE_SET - set(field_cols))
    if loss_detail_cols is None:
        loss_detail_cols = []
    if not all([col in detail_cols for col in loss_detail_cols]):
        raise Exception("loss_detail_cols must be a subset of detail_cols")
    if collapse_fields is None:
        collapse_fields = []

    # noinspection DuplicatedCode
    if metadata is None:
        metadata = Metadata()
    cell_metadata = _create_metadata(df, metadata, detail_cols, loss_detail_cols)
    period_starts = list(df.period_start.dt.date)
    period_ends = list(df.period_end.dt.date)
    evaluation_dates = list(df.evaluation_date.dt.date)
    if "prev_evaluation_date" in df.columns:
        prev_evaluation_dates = list(df.prev_evaluation_date.dt.date)

    raw_values = {colname: _clean_list_column(df[colname]) for colname in field_cols}
    values = [
        {k: np.array(v[i]) for k, v in raw_values.items() if v[i] is not None}
        for i in range(df.shape[0])
    ]

    if "prev_evaluation_date" in df.columns:
        cells = [
            IncrementalCell(
                period_start=period_start,
                period_end=period_end,
                evaluation_date=evaluation_date,
                prev_evaluation_date=prev_evaluation_date,
                metadata=metadata,
                values=value,
            )
            for (
                metadata,
                period_start,
                period_end,
                evaluation_date,
                prev_evaluation_date,
                value,
            ) in zip(
                cell_metadata,
                period_starts,
                period_ends,
                evaluation_dates,
                prev_evaluation_dates,
                values,
            )
        ]
    else:
        if "risk_basis" not in df.columns:
            df = df.assign(risk_basis=[cell.risk_basis for cell in cell_metadata])
        if "per_occurrence_limit" not in df.columns:
            df = df.assign(
                per_occurrence_limit=[
                    cell.per_occurrence_limit for cell in cell_metadata
                ]
            )
        cells = []
        for idx, group_df in df.groupby(
            [
                "period_start",
                "period_end",
                "evaluation_date",
                "risk_basis",
                "per_occurrence_limit",
            ]
            + detail_cols
            + loss_detail_cols,
            dropna=False,
        ):
            period_start, period_end, evaluation_date, *_ = idx
            current_metadata = _create_metadata(
                group_df, metadata, detail_cols, loss_detail_cols
            )[0]
            if len(group_df) > 1:
                try:
                    group_df = group_df.sort_values(["scenario"])
                except KeyError:
                    raise Exception(
                        "Multiple values found for this Cell. If multiple values are present for a single field they must be identified by a `scenario` column."
                    )
            values = {
                colname: _clean_list_column(group_df[colname]) for colname in field_cols
            }

            # Check for scalar values, typically earned_premium
            for field in collapse_fields:
                if all(val == values[field][0] for val in values[field]):
                    values[field] = values[field][0]
                else:
                    raise warn(
                        f"Field {field} has multiple distinct values, will not be collapsed"
                    )

            # coerce values types
            for field, vals in values.items():
                if isinstance(vals, list) and len(vals) > 1:
                    values[field] = np.array(vals)
                if isinstance(vals, list) and len(vals) == 1:
                    values[field] = vals[0]
            values = tlz.valfilter(lambda val: val is not None, values)

            cells.append(
                CumulativeCell(
                    period_start=period_start.date(),
                    period_end=period_end.date(),
                    evaluation_date=evaluation_date.date(),
                    metadata=current_metadata,
                    values=values,
                )
            )

    return Triangle(cells)


def long_data_frame_to_triangle(
    df: pd.DataFrame,
    loss_detail_cols: list[str] | None = None,
    metadata: Metadata | None = None,
) -> Union[Triangle[CumulativeCell], Triangle[IncrementalCell]]:
    """
    Convert a long-format `pd.DataFrame` to a `Triangle`.

    A long-format `DataFrame` requires the following columns to be present: `period_start`,
    `period_end`, `evaluation_date`, `field` and `value`. When the column 'prev_evaluation_date'
    is present, a triangle with incremental cells is generated. Otherwise, all triangle cells
    are cumulative.

    Each row of the `DataFrame` represents a field of a single `Cell` of the `Triangle`.
    Any columns besides the three required columns are interpreted as metadata
    (i.e., `per_occurrence_limit`, `currency`, `program_name`).

    The `metadata` argument can be used to supply common metadata that may not be directly
    present in the `DataFrame`. For example, every `Cell` in the `Triangle` could be
    accident-basis, gross of reinsurance, including DCC. This could be supplied by a
    single `Metadata` object in the function call instead of adding several constant columns
    to the `DataFrame`.

    Args:
        df: Pandas `DataFrame` to convert to a `Triangle`.
        loss_detail_cols: List of columns that belong to `loss_details` in `Metadata`, rather
            than `details`.
        metadata: Default metadata to apply to every cell in the `Triangle`.
    """
    _check_index_columns(df)
    if "field" not in df:
        raise Exception(
            "Long DataFrame must have a column `field` to convert to a Triangle"
        )
    if not pd.api.types.is_string_dtype(df.field):
        raise Exception("Long DataFrame field column must be a string")
    if "value" not in df:
        raise Exception(
            "Long DataFrame must have a column `value` to convert to a Triangle"
        )
    if not pd.api.types.is_numeric_dtype(df.value):
        raise Exception("Long DataFrame value column must be numeric")

    if loss_detail_cols is None:
        loss_detail_cols = []
    detail_cols = list(
        set(df.columns.values) - CORE_SET - {"field", "value"} - set(loss_detail_cols)
    )

    # noinspection DuplicatedCode
    if metadata is None:
        metadata = Metadata()
    cell_metadata = _create_metadata(df, metadata, detail_cols, loss_detail_cols)
    period_starts = list(df.period_start.dt.date)
    period_ends = list(df.period_end.dt.date)
    evaluation_dates = list(df.evaluation_date.dt.date)
    if "prev_evaluation_date" in df.columns:
        prev_evaluation_dates = list(df.prev_evaluation_date.dt.date)
    fields = list(df.field)
    values = list(df.value)

    cells = {}
    if "prev_evaluation_date" in df.columns:
        for (
            metadata,
            period_start,
            period_end,
            evaluation_date,
            prev_evaluation_date,
            field,
            value,
        ) in zip(
            cell_metadata,
            period_starts,
            period_ends,
            evaluation_dates,
            prev_evaluation_dates,
            fields,
            values,
        ):
            index = (
                period_start,
                period_end,
                evaluation_date,
                prev_evaluation_date,
                metadata,
            )
            if value is not None:
                if index in cells:
                    if field in cells[index].values:
                        raise Exception(
                            f"Field {field} is already present in this `Cell`"
                        )
                    cells[index].values[field] = np.array(value)
                else:
                    cells[index] = IncrementalCell(
                        period_start=period_start,
                        period_end=period_end,
                        evaluation_date=evaluation_date,
                        prev_evaluation_date=prev_evaluation_date,
                        metadata=metadata,
                        values={field: np.array(value)},
                    )
    else:
        if "risk_basis" not in df.columns:
            df = df.assign(risk_basis=[cell.risk_basis for cell in cell_metadata])
        if "per_occurrence_limit" not in df.columns:
            df = df.assign(
                per_occurrence_limit=[
                    cell.per_occurrence_limit for cell in cell_metadata
                ]
            )
        for idx, group_df in df.groupby(
            [
                "period_start",
                "period_end",
                "evaluation_date",
                "field",
                "risk_basis",
                "per_occurrence_limit",
            ]
            + detail_cols
            + loss_detail_cols,
            dropna=False,
        ):
            period_start, period_end, evaluation_date, field, *_ = idx
            current_metadata = _create_metadata(
                group_df, metadata, detail_cols, loss_detail_cols
            )[0]
            index = (
                period_start,
                period_end,
                evaluation_date,
                current_metadata,
            )
            if len(group_df) > 1:
                try:
                    group_df = group_df.sort_values(["scenario"])
                except KeyError:
                    raise Exception(
                        f"Field {field} is already present in this Cell. If multiple values are present for a single field they must be identified by a `scenario` column."
                    )
            value = group_df["value"].values
            # Check for scalar values, typically earned_premium
            if len(group_df) == 1 or (
                "scenario" in group_df and all(pd.isnull(group_df["scenario"]))
            ):
                value = value[0]
            if value is not None:
                if index in cells:
                    if field in cells[index].values:
                        raise Exception(
                            f"Field {field} is already present in this `Cell`"
                        )
                    cells[index].values[field] = value
                else:
                    cells[index] = CumulativeCell(
                        period_start=period_start.date(),
                        period_end=period_end.date(),
                        evaluation_date=evaluation_date.date(),
                        metadata=current_metadata,
                        values={field: value},
                    )
    return Triangle([cell for cell in cells.values()])


def _check_index_columns(df: pd.DataFrame) -> None:
    for column in INDEX_CUM_COLUMNS:
        if column not in df:
            raise Exception(
                f"DataFrame must have a column `{column}` to convert to a Triangle"
            )
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            if df[column].map(lambda i: isinstance(i, datetime.date)).all():
                df[column] = pd.to_datetime(df[column])
            else:
                raise Exception(f"DataFrame column {column} must be a datetime")
    if "prev_evaluation_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["prev_evaluation_date"]):
            raise Exception(
                "DataFrame column `prev_evaluation_date` must be a datetime"
            )


def _create_metadata(
    df: pd.DataFrame,
    metadata: Metadata,
    detail_columns: list[str],
    loss_detail_columns: list[str],
) -> list[Metadata]:
    risk_bases = _clean_metadata_column(df, "risk_basis", metadata.risk_basis)
    countries = _clean_metadata_column(df, "country", metadata.country)
    currencies = _clean_metadata_column(df, "currency", metadata.currency)
    reinsurance_bases = _clean_metadata_column(
        df, "reinsurance_basis", metadata.reinsurance_basis
    )
    loss_definitions = _clean_metadata_column(
        df, "loss_definition", metadata.loss_definition
    )
    per_occurrence_limits = _clean_metadata_column(
        df, "per_occurrence_limit", metadata.per_occurrence_limit
    )

    # Build details and loss_details columns
    pure_detail_columns = list(set(detail_columns) - set(loss_detail_columns))
    details_list = _create_metadata_details(df, pure_detail_columns, metadata.details)
    loss_details_list = _create_metadata_details(
        df, loss_detail_columns, metadata.loss_details
    )

    return [
        Metadata(
            risk_basis=risk_basis,
            country=country,
            currency=currency,
            reinsurance_basis=reinsurance_basis,
            loss_definition=loss_definition,
            per_occurrence_limit=per_occurrence_limit,
            details=details,
            loss_details=loss_details,
        )
        for (
            risk_basis,
            country,
            currency,
            reinsurance_basis,
            loss_definition,
            per_occurrence_limit,
            details,
            loss_details,
        ) in zip(
            risk_bases,
            countries,
            currencies,
            reinsurance_bases,
            loss_definitions,
            per_occurrence_limits,
            details_list,
            loss_details_list,
        )
    ]


def _create_metadata_details(
    df: pd.DataFrame,
    detail_columns: list[str],
    default_details: dict[str, Any],
) -> list[dict[str, Any]]:
    # Load up the details from the dataframe
    details = [{} for _ in range(df.shape[0])]
    for colname in detail_columns:
        clean_column = _clean_list_column(df[colname], default_details.get(colname))
        for i, val in enumerate(clean_column):
            if val is not None:
                details[i][colname] = val

    # Load in columns in the defaults but not in the dataframe
    default_cols = {col for col in default_details if col not in detail_columns}
    for colname in default_cols:
        for i in range(len(details)):
            details[i][colname] = default_details[colname]

    return details


def _clean_metadata_column(df: pd.DataFrame, colname: str, default: Any) -> list[Any]:
    if colname in df:
        return _clean_list_column(df[colname], default)
    else:
        return [default for _ in range(df.shape[0])]


def _clean_list_column(column: pd.Series, default: Any = None) -> list[Any]:
    return [default if missing else e for e, missing in zip(column, column.isna())]
