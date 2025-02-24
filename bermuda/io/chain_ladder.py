from __future__ import annotations

import datetime

from ..base import CumulativeCell, Metadata
from ..date_utils import add_months
from ..triangle import Triangle
from .data_frame_output import triangle_to_wide_data_frame


def triangle_to_chain_ladder(
    triangle: Triangle,
) -> chainladder.core.triangle.Triangle:  # noqa: F821
    """Convert a Triangle to a chainladder Triangle object.

    Args:
        triangle (Triangle): The Bermuda Triangle to convert.

    Returns:
        chainladder.core.triangle.Triangle: The chainladder Triangle object.
    """
    try:
        import chainladder as cl
    except ImportError:
        raise ImportError(
            "The chainladder package is required to convert a Triangle to a ChainLadder object"
        )
    df = triangle_to_wide_data_frame(triangle)
    index = set(df.columns) - set(
        triangle.fields + ["period_start", "period_end", "evaluation_date"]
    )
    cl_triangle = cl.Triangle(
        df,
        origin="period_start",
        development="evaluation_date",
        columns=triangle.fields,
        origin_format="%Y-%m-%d",
        development_format="%Y-%m-%d",
        cumulative=not triangle.is_incremental,
        index=list(index),
    )
    return cl_triangle


def chain_ladder_to_triangle(
    chain_ladder: chainladder.core.triangle.Triangle,  # noqa: F821
    base_metadata: Metadata | None = None,
) -> Triangle:
    """Create a Triangle from a chainladder Triangle object.

    Anything in the chainladder index will become details in the Bermuda Triangle.

    Args:
        chain_ladder (chainladder.core.triangle.Triangle): The chainladder Triangle object.
        base_metadata (Metadata): The base metadata to apply to all cells. Defaults to None.

    Returns:
        Triangle: The Bermuda Triangle.
    """
    if base_metadata is None:
        base_metadata = Metadata()
    if not chain_ladder.is_cumulative:
        raise ValueError("Only cumulative ChainLadder triangles are supported")
    cl_df = chain_ladder.val_to_dev().to_frame()
    origin_resolution = {"Y": 12, "Q": 3, "M": 1}[chain_ladder.origin_grain]
    if chain_ladder.shape[0] == 1:
        df_long = cl_df.reset_index().melt(
            id_vars="index", var_name="dev_lag", value_name="values"
        )
        cells = []
        for _, row in df_long.dropna(subset=["values"]).iterrows():
            period_end = add_months(
                row["index"], origin_resolution
            ) - datetime.timedelta(days=1)

            cell_values = {chain_ladder.columns[0]: row["values"]}

            cell = CumulativeCell(
                period_start=row["index"].date(),
                period_end=period_end,
                evaluation_date=add_months(
                    period_end, row["dev_lag"] - origin_resolution
                ),
                values=cell_values,
                metadata=base_metadata,
            )
            cells.append(cell)
        return Triangle(cells)
    # group by index and iterate over groups
    index_names = chain_ladder.key_labels
    cells = []
    for group_id, group_df in cl_df.groupby(level=index_names):
        details = {k: v for k, v in zip(index_names, group_id)}
        for _, row in group_df.iterrows():
            period_end = add_months(
                row["origin"], origin_resolution
            ) - datetime.timedelta(days=1)

            cell_values = row.drop(["origin", "development"]).to_dict()

            cell = CumulativeCell(
                period_start=row["origin"].date(),
                period_end=period_end,
                evaluation_date=add_months(
                    period_end, row["development"] - origin_resolution
                ),
                values=cell_values,
                metadata=Metadata(**{**base_metadata.as_dict(), "details": details}),
            )
            cells.append(cell)

    return Triangle(list(cells))
