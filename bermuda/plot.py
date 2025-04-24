import altair as alt
from babel.numbers import get_currency_symbol
import pandas as pd
from math import ceil, exp

from .triangle import Triangle

alt.renderers.enable("browser")

SLICE_TITLE_KWARGS = {
    "anchor": "middle",
    "font": "monospace",
    "fontWeight": "normal",
    "fontSize": 12,
}

BASE_HEIGHT = 600
BASE_WIDTH = "container"

BASE_AXIS_LABEL_FONT_SIZE = 16
BASE_AXIS_TITLE_FONT_SIZE = 18
SIZE_FACTOR = 0.1

@alt.theme.register("bermuda_plot_theme", enable=True)
def bermuda_plot_theme() -> alt.theme.ThemeConfig:
    return {
        "autosize": {"contains": "content", "resize": True},
        "config": {
            "style": {
                "group-title": {"fontSize": 24},
                "group-subtitle": {"fontSize": 18},
                "guide-label": {"fontSize": BASE_AXIS_LABEL_FONT_SIZE, "font": "monospace"},
                "guide-title": {"fontSize": BASE_AXIS_TITLE_FONT_SIZE, "font": "monospace"},
            },
            "mark": {"color": "black"},
            "title": {"anchor": "start"},
            "legend": {
                "orient": "right", 
                "titleAnchor": "start",
                "layout": {
                    "direction": "vertical",
                },
            },
        },
    }

def _currency_symbol(triangle: Triangle) -> str:
    code = triangle.metadata[0].currency
    return get_currency_symbol(code, locale="en_US") or "$"

def _concat_slice_charts(charts: list[alt.Chart], **kwargs) -> alt.Chart:
    if len(charts) == 1:
        return charts[0].properties(**kwargs)

    ncols = max(2, ceil(len(charts) / 4))
    fig = alt.concat(*charts, autosize="pad", columns=ncols, **kwargs)
    return fig


def plot_right_edge(triangle: Triangle) -> alt.Chart:
    main_title = alt.Title(
        "Latest Loss Ratio",
        subtitle="The most recent loss ratio diagonal"
    )
    width = BASE_WIDTH if len(triangle.slices) == 1 else 400 * 2 / len(triangle.slices)
    height = BASE_HEIGHT if len(triangle.slices) == 1 else 200 * 2 / len(triangle.slices)
    n_slices = len(triangle.slices)
    fig = _concat_slice_charts(
        [
            plot_right_edge_slice(
                triangle_slice, 
                alt.Title(f"slice {i + 1}", **SLICE_TITLE_KWARGS),
                n_slices,
            ).properties(width=width, height=height) for i, (m, triangle_slice) 
            in enumerate(triangle.slices.items())
        ],
        title=main_title,
    )
    return fig.configure_axis(
        **_compute_font_sizes(n_slices),
    ).configure_legend(
        **_compute_font_sizes(n_slices),
    )

def plot_right_edge_slice(triangle: Triangle, title: alt.Title, n_slices: int) -> alt.Chart:
    if not (
        "earned_premium" in triangle.fields
        and ("paid_loss" in triangle.fields or "reported_loss" in triangle.fields)
    ):
        raise ValueError(
            "Triangle must contain earned_premium and either paid or reported loss fields "
            f"in order to plot right edge. This triangle contains {triangle.fields}"
        )


    loss_fields = [field for field in triangle.fields if "_loss" in field]

    loss_data = alt.Data(values=[
        *[{
            "period_start": pd.to_datetime(cell.period_start),
            "period_end": pd.to_datetime(cell.period_end),
            "evaluation_date": pd.to_datetime(cell.evaluation_date),
            "dev_lag": cell.dev_lag(),
            "Loss Ratio": (
                cell[field] / cell["earned_premium"]
                if field in cell
                else None
            ),
            "Field": field.replace("_", " ").title(),
        }
        for cell in triangle.right_edge
        for field in loss_fields
        if "earned_premium" in cell
        ]
    ])

    premium_data = alt.Data(values=[
        *[{
            "period_start": pd.to_datetime(cell.period_start),
            "period_end": pd.to_datetime(cell.period_end),
            "evaluation_date": pd.to_datetime(cell.evaluation_date),
            "dev_lag": cell.dev_lag(),
            "Earned Premium": cell["earned_premium"],
            "Field": "Earned Premium",
        }
        for cell in triangle.right_edge
        if "earned_premium" in cell
        ]
    ])

    currency = _currency_symbol(triangle)

    selection = alt.selection_point()

    bar = alt.Chart(premium_data, title=title).mark_bar().encode(
        x=alt.X(f"yearmonth(period_start):O"),
        y=alt.Y("Earned Premium:Q"),
        color=alt.Color("Field:N").scale(range=["lightgray"]),
        tooltip=[alt.Tooltip("period_start:T", title="Period Start"), alt.Tooltip("period_end:T", title="Period End"), alt.Tooltip("dev_lag:O", title="Dev Lag"), alt.Tooltip("evaluation_date:T", title="Evaluation Date"), alt.Tooltip("Earned Premium:Q", format=f"{currency},.0f")],
    )

    lines = alt.Chart(loss_data).mark_line(
        size=1,
    ).encode(
        x=alt.X(f"yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title("Period Start"),
        y=alt.Y("Loss Ratio:Q", scale=alt.Scale(zero=True), axis=alt.Axis(format="%")).title("Loss Ratio (%)"),
        color=alt.Color("Field:N", legend=None),
    )

    points = alt.Chart(loss_data).mark_point(
        size=max(20, 100 * 1 / n_slices),
        filled=True,
        opacity=1,
    ).encode(
        x=alt.X(f"yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title("Period Start"),
        y=alt.Y("Loss Ratio:Q", scale=alt.Scale(zero=True), axis=alt.Axis(format="%")).title("Loss Ratio (%)"),
        color=alt.Color("Field:N").legend(title=None),
        tooltip=[alt.Tooltip("period_start:T", title="Period Start"), alt.Tooltip("period_end:T", title="Period End"), alt.Tooltip("dev_lag:O", title="Dev Lag"), alt.Tooltip("evaluation_date:T", title="Evaluation Date"), alt.Tooltip("Loss Ratio:Q", format=".2%")],
    )

    fig = alt.layer(bar, lines, points).resolve_scale(
        y="independent",
        color="independent",
    )

    return fig.interactive()


def plot_data_completeness(triangle: Triangle) -> alt.Chart:
    main_title = alt.Title(
        "Triangle Completeness",
        subtitle="The number of fields available per cell",
    )
    width = BASE_WIDTH if len(triangle.slices) == 1 else 400 * 3 / len(triangle.slices)
    height = BASE_HEIGHT if len(triangle.slices) == 1 else 300  * 3 / len(triangle.slices)
    n_slices = len(triangle.slices)
    fig = _concat_slice_charts(
        [
            plot_data_completeness_slice(
                triangle_slice, 
                alt.Title(f"slice {i + 1}", **SLICE_TITLE_KWARGS),
                n_slices,
            ).properties(width=width, height=height) for i, (_, triangle_slice) 
            in enumerate(triangle.slices.items())
        ],
        title=main_title,
    ).configure_axis(
        **_compute_font_sizes(n_slices),
    )
    return fig


def plot_data_completeness_slice(triangle: Triangle, title: alt.Title, n_slices: int) -> alt.Chart:
    if not triangle.is_disjoint:
        raise Exception(
            "This triangle isn't disjoint! You probably don't want to use it"
        )
    if not triangle.is_semi_regular:
        raise Exception(
            "This triangle isn't semi-regular! You probably don't want to use it"
        )

    currency = _currency_symbol(triangle)
    
    cell_data = alt.Data(values=[
        *[
            {
                "period_start": pd.to_datetime(cell.period_start), 
                "period_end": pd.to_datetime(cell.period_end), 
                "evaluation_date": pd.to_datetime(cell.evaluation_date), 
                "dev_lag": cell.dev_lag(), 
                "Number of Fields": len(cell.values), 
                "Fields": ", ".join([field.replace("_", " ").title() + f" ({currency}{cell[field]:,.0f})" for field in cell.values]),
            } for cell in triangle.cells
        ]
    ])

    selection = alt.selection_point()

    fig = alt.Chart(
        cell_data, 
        title=title,
    ).mark_circle(size=500 * 1 / n_slices, opacity=1).encode(
        alt.X("dev_lag:N", axis=alt.Axis(labelAngle=0), scale=alt.Scale(zero=True)).title("Dev Lag (months)"),
        alt.Y("yearmonth(period_start):T", scale=alt.Scale(padding=15, reverse=True)).title("Period Start"), 
        color=alt.condition(selection, alt.Color("Number of Fields:N").scale(scheme="dark2"), alt.value("lightgray")),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:N", title="Dev Lag (months)"),
            alt.Tooltip("Fields:N"), 
        ],
    ).add_params(
        selection
    )

    return fig.interactive()


def _compute_font_sizes(n_slices: int) -> dict[str, float | int]:
    return {
        "titleFontSize": BASE_AXIS_TITLE_FONT_SIZE * exp(-SIZE_FACTOR * (n_slices - 1)),
        "labelFontSize": BASE_AXIS_LABEL_FONT_SIZE * exp(-SIZE_FACTOR * (n_slices - 1)) 
    }

