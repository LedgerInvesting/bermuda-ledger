import datetime
import re
from warnings import warn

import pandas as pd

from ..base import CumulativeCell, Metadata
from ..date_utils import add_months, calculate_dev_lag
from ..triangle import Triangle
from ..utils import merge


def array_triangle_builder(
    dfs: list[pd.DataFrame],
    fields: list[str],
    **kwargs,
) -> Triangle:
    """
    Build a Triangle object from a list of DataFrames in a triangular array format.

    Args:
        dfs: List of DataFrames in a triangular array format.
        fields: List of field names.
        **kwargs: Additional arguments to pass to array_data_frame_to_triangle.
    """
    if not len(dfs) == len(fields):
        raise ValueError("The number of DataFrames and fields must match.")
    triangle = array_data_frame_to_triangle(dfs[0], fields[0], **kwargs)
    for df, field in zip(dfs[1:], fields[1:]):
        triangle = merge(triangle, array_data_frame_to_triangle(df, field, **kwargs))
    return triangle


def array_data_frame_to_triangle(
    df: pd.DataFrame,
    field: str,
    period_resolution: int | None = None,
    eval_resolution: int | None = None,
    dev_lag_from_period_end: bool = True,
    metadata: Metadata | None = None,
) -> Triangle:
    """
    Convert a DataFrame in a triangular array format to a Triangle object. The first
    column is assumed to contain the accident or policy period. This should be a date,
    or a string that can be parsed as a date (e.g. "2020", or "2020Q1"). All subsequent
    columns are assumed to contain values for the development periods. If integer development
    lags cannot be inferred from the column names an eval_resolution should be provided.

    Args:
        df: DataFrame in a triangular array format.
        field: Field name.
        period_resolution: Period resolution in months. If None, this is inferred from
            the first consecutive period dates.
        eval_resolution: Evaluation resolution in months. If None, this is inferred from
            the column names. If None and column names are missing, it's assumed to be
            the same as period_resolution.
        dev_lag_from_period_end: Is the development lag calculated from the period_end
            rather than the period start? The bermuda convention is to calculate dev
            lags as the number of months from the end of the period (so the first natural
            evaluation for any period is lag 0). If False, the dev lag is calculated from
            the period start.
        metadata: Metadata associated with the triangle.
    """

    df = df.rename(columns={df.columns[0]: "period"})
    df["period"] = df["period"].apply(parse_date)
    if period_resolution is None:
        if df.shape[0] < 2:
            raise ValueError(
                "At least two periods are required to infer period resolution."
            )
        period_resolution = int(
            calculate_dev_lag(df["period"].iloc[0], df["period"].iloc[1])
        )
    if eval_resolution is None:
        try:
            [int(x) for x in df.columns[1:]]
        except ValueError:
            warn(
                (
                    "Could not infer evaluation resolution from column names - "
                    f"defaulting to period resolution {period_resolution}"
                )
            )
            eval_resolution = period_resolution

    if metadata is None:
        metadata = Metadata()

    cells = []
    for _, row in df.iterrows():
        period_start = row["period"]
        period_end = add_months(row["period"], period_resolution) - datetime.timedelta(
            days=1
        )
        current_lag = 0
        for dev_lag in df.columns[1:]:
            if eval_resolution is None:
                current_lag = int(dev_lag)
            value = row[dev_lag]
            if dev_lag_from_period_end or eval_resolution is not None:
                # Typical case where we're using the bermuda dev lag convention
                # or just inferring the eval resolution from the period resolution
                evaluation_date = add_months(period_end, current_lag)
            else:
                # If specified by the user we can handle dev lags using the
                # period_start as 0
                evaluation_date = add_months(
                    period_start, current_lag
                ) - datetime.timedelta(days=1)
            if not pd.isna(value):
                cells.append(
                    CumulativeCell(
                        period_start=period_start,
                        period_end=period_end,
                        evaluation_date=evaluation_date,
                        values={field: row[dev_lag]},
                        metadata=metadata,
                    )
                )
            if eval_resolution is not None:
                current_lag += eval_resolution
    return Triangle(cells)


def statics_data_frame_to_triangle(
    df: pd.DataFrame,
    evaluation_date: datetime.date | None = None,
    period_resolution: int | None = None,
    metadata: Metadata | None = None,
):
    """
    Convert a DataFrame in a "statics" format to a Triangle object. The first
    column is assumed to contain the accident or policy period. This should be a date,
    or a string that can be parsed as a date (e.g. "2020", or "2020Q1"). All subsequent
    columns are assumed to contain values for a single development period that are typically
    assumed to be static over development periods (e.g. earned premium, earned exposure).
    In order to convert these values into a triangle an evaluation_date must be provided. If
    the resulting triangle can often be added to an existing loss triangle using the add_statics
    function, in which case the evaluation_date is just a placeholder (though this must be greater
    than the latest period). If none is provided we will use the last period end as the evaluation
    date. The period resolution can be inferred from the first two consecutive periods, otherwise
    it must be provided as an integer. Metadata can be provided to add additional information to
    the cells. Field names will be inferred from the column names.

    Args:
        df: DataFrame to convert.
        evaluation_date: Evaluation date for the static values. This is typically a placeholder
            date that is greater than the latest period in the triangle.
        period_resolution: Period resolution in months. If None, this is inferred from
            the first consecutive period dates.
        metadata: Metadata associated with the triangle.
    """
    df = df.rename(columns={df.columns[0]: "period"})
    df["period"] = df["period"].apply(parse_date)
    if period_resolution is None:
        if df.shape[0] < 2:
            raise ValueError(
                "At least two periods are required to infer period resolution."
            )
        period_resolution = (df["period"].iloc[1] - df["period"].iloc[0]).days // 30
    if evaluation_date is None:
        evaluation_date = add_months(
            df["period"].max(), period_resolution
        ) - datetime.timedelta(days=1)
    if metadata is None:
        metadata = Metadata()

    cells = []
    for _, row in df.iterrows():
        period_start = row["period"]
        period_end = add_months(row["period"], period_resolution) - datetime.timedelta(
            days=1
        )
        cells.append(
            CumulativeCell(
                period_start=period_start,
                period_end=period_end,
                evaluation_date=evaluation_date,
                values=row[1:].to_dict(),
                metadata=metadata,
            )
        )
    return Triangle(cells)


def parse_date(value):
    """Convert various date formats into datetime.date objects."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        pass

    if re.fullmatch(r"\d{4}", value):
        return datetime.date(int(value), 1, 1)

    match = re.fullmatch(r"(\d{4})Q([1-4])", value)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime.date(year, month, 1)

    match = re.fullmatch(r"(\d{4})H([1-2])", value)
    if match:
        year, half = int(match.group(1)), int(match.group(2))
        month = 1 if half == 1 else 7
        return datetime.date(year, month, 1)
    raise ValueError(f"Could not parse date: {value}")


def triangle_to_array_data_frame(triangle: Triangle, field: str):
    """
    Convert a Triangle object to a DataFrame in a triangular array format.

    Args:
        triangle: A single-sliced Triangle object.
        field: Field name.
    """
    filtered_tri = triangle.filter(lambda cell: field in cell.values)
    if len(triangle.slices) > 1:
        raise ValueError("Triangle must be single-sliced.")
    if triangle.is_incremental:
        raise ValueError("Triangle must be cumulative.")
    rows = []
    for period, cells in filtered_tri.period_rows:
        row = {"period": period[0]}
        for cell in cells:
            row[str(int(cell.dev_lag()))] = cell[field]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def triangle_to_right_edge_data_frame(triangle: Triangle) -> pd.DataFrame:
    """
    Convert a Triangle object to a DataFrame displaying values
    on the right edge of the triangle. The resulting dataframe will
    show the 'period', 'evaluation_date', and cell values for the
    right edge of the triangle.

    Args:
        triangle: A single-sliced Triangle object.
    """
    if len(triangle.slices) > 1:
        raise ValueError("Triangle must be single-sliced.")
    if triangle.is_incremental:
        raise ValueError("Triangle must be cumulative.")
    rows = [
        {
            "period": cell.period_start,
            "evaluation_date": cell.evaluation_date,
            **cell.values,
        }
        for cell in triangle.right_edge
    ]
    return pd.DataFrame(rows)
