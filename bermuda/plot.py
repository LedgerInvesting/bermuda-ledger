import altair as alt
from babel.numbers import get_currency_symbol
import pandas as pd
import numpy as np

from .triangle import Triangle, Cell

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
FONT_SIZE_DECAY_FACTOR = 0.1

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
            "title": {"anchor": "start", "offset": 20},
            "legend": {
                "orient": "right", 
                "titleAnchor": "start",
                "layout": {
                    "direction": "vertical",
                },
            },
        },
    }


def plot_right_edge(triangle: Triangle, show_uncertainty: bool = True, uncertainty_type: str = "ribbon") -> alt.Chart:
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
                show_uncertainty,
                uncertainty_type,
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


def plot_right_edge_slice(triangle: Triangle, title: alt.Title, n_slices: int, show_uncertainty: bool = True, uncertainty_type: str = "ribbon") -> alt.Chart:
    if not "earned_premium" in triangle.fields:
        raise ValueError(
            "Triangle must contain `earned_premium` to plot its right edge. "
            f"This triangle contains {triangle.fields}"
        )

    loss_fields = [field for field in triangle.fields if "_loss" in field]

    loss_data = alt.Data(values=[
        *[{
            "period_start": pd.to_datetime(cell.period_start),
            "period_end": pd.to_datetime(cell.period_end),
            "evaluation_date": pd.to_datetime(cell.evaluation_date),
            "dev_lag": cell.dev_lag(),
            **_calculate_field_summary(cell, lambda ob: ob[field] / ob["earned_premium"], "loss_ratio"),
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

    if show_uncertainty and uncertainty_type == "ribbon":
        loss_error = alt.Chart(loss_data).mark_area(
            opacity=0.5,
        ).encode(
            x=alt.X(f"yearmonth(period_start):T"),
            y=alt.Y("loss_ratio_lower_ci:Q").axis(title=None, format="%"),
            y2=alt.Y2("loss_ratio_upper_ci:Q"),
            color=alt.Color("Field:N"),
        )
    elif show_uncertainty and uncertainty_type == "segments":
        loss_error = alt.Chart(loss_data).mark_errorbar(
        ).encode(
            x=alt.X(f"yearmonth(period_start):T"),
            y=alt.Y("loss_ratio_lower_ci:Q").axis(title=None, format="%"),
            y2=alt.Y2("loss_ratio_upper_ci:Q"),
            color=alt.Color("Field:N"),
        )
    else:
        loss_error = alt.LayerChart()

    lines = alt.Chart(loss_data).mark_line(
        size=1,
    ).encode(
        x=alt.X(f"yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title("Period Start"),
        y=alt.Y("loss_ratio:Q", scale=alt.Scale(zero=True), axis=alt.Axis(format="%")).title("Loss Ratio (%)"),
        color=alt.Color("Field:N"),
    )

    points = alt.Chart(loss_data).mark_point(
        size=max(20, 100 * 1 / n_slices),
        filled=True,
        opacity=1,
    ).encode(
        x=alt.X(f"yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title("Period Start"),
        y=alt.Y("loss_ratio:Q", scale=alt.Scale(zero=True), axis=alt.Axis(format="%")).title("Loss Ratio (%)"),
        color=alt.Color("Field:N").legend(title=None),
        tooltip=[alt.Tooltip("period_start:T", title="Period Start"), alt.Tooltip("period_end:T", title="Period End"), alt.Tooltip("dev_lag:O", title="Dev Lag"), alt.Tooltip("evaluation_date:T", title="Evaluation Date"), alt.Tooltip("loss_ratio:Q", title="Loss Ratio (%)", format=".2%")],
    )

    fig = alt.layer(bar, loss_error + lines + points).resolve_scale(
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

    selection = alt.selection_point()
    
    cell_data = alt.Data(values=[
        *[
            {
                "period_start": pd.to_datetime(cell.period_start), 
                "period_end": pd.to_datetime(cell.period_end), 
                "evaluation_date": pd.to_datetime(cell.evaluation_date), 
                "dev_lag": cell.dev_lag(), 
                "Number of Fields": len(cell.values), 
                "Fields": ", ".join([field.replace("_", " ").title() + f" ({currency}{np.mean(cell[field]):,.0f})" for field in cell.values]),
            } for cell in triangle.cells
        ]
    ])

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


def plot_heatmap(triangle: Triangle, field: str = "paid_loss") -> alt.Chart:
    main_title = alt.Title(
        "Triangle Loss Ratio Heatmap",
    )
    width = BASE_WIDTH if len(triangle.slices) == 1 else 400 * 3 / len(triangle.slices)
    height = BASE_HEIGHT if len(triangle.slices) == 1 else 300  * 3 / len(triangle.slices)
    n_slices = len(triangle.slices)
    fig = _concat_slice_charts(
        [
            plot_heatmap_slice(
                triangle_slice, 
                field,
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

def plot_heatmap_slice(triangle: Triangle, field: str = "paid_loss", title: alt.Title = alt.Title(""), n_slices: int = 1) -> alt.Chart:
    if not "earned_premium" in triangle.fields:
        raise ValueError(
            "Triangle must contain `earned_premium` to plot the loss ratio heatmap. "
            f"This triangle contains {triangle.fields}"
        )

    loss_data = alt.Data(values=[
        *[{
            "period_start": pd.to_datetime(cell.period_start),
            "period_end": pd.to_datetime(cell.period_end),
            "evaluation_date": pd.to_datetime(cell.evaluation_date),
            "dev_lag": cell.dev_lag(),
            **_calculate_field_summary(cell, lambda ob: 100 * ob[field] / ob["earned_premium"], "loss_ratio"),
            "Field": field.replace("_", " ").title(),
        }
        for cell in triangle
        if "earned_premium" in cell
        ]
    ])


    base = alt.Chart(loss_data, title=title).encode(
        x=alt.X("dev_lag:N", axis=alt.Axis(labelAngle=0)).title("Dev Lag (months)"),
        y=alt.X("yearmonth(period_start):O", scale=alt.Scale(reverse=False)).title("Period Start"),
    )

    
    stroke_predicate = alt.datum.loss_ratio_sd / alt.datum.loss_ratio > 0
    selection = alt.selection_interval()
    heatmap = base.mark_rect(
    ).encode(
        color=alt.when(
            selection
        ).then(alt.Color("loss_ratio:Q").title(field.replace("_", " ").title() + " %")).otherwise(
            alt.value("gray"),
        ),
        tooltip=[alt.Tooltip("period_start:T", title="Period Start"), alt.Tooltip("period_end:T", title="Period End"), alt.Tooltip("evaluation_date:T", title="Evaluation Date"), alt.Tooltip("dev_lag:O", title="Dev Lag (months)")],
        stroke=alt.when(stroke_predicate).then(alt.value("black")),
        strokeWidth=alt.when(stroke_predicate).then(alt.value(3)).otherwise(alt.value(0)),
    ).add_params(
        selection
    )

    text_color_predicate = alt.datum.loss_ratio > 70
    text = base.mark_text(fontSize=20 * np.exp(-FONT_SIZE_DECAY_FACTOR * n_slices), font="monospace").encode(
        text=alt.Text("loss_ratio:Q", format=".2f"),
        color=alt.when(text_color_predicate).then(alt.value("lightgray")).otherwise(alt.value("black")),
    )

    return heatmap + text


def _calculate_field_summary(cell: Cell, fn: callable, name: str, probs: tuple[float, float] = (0.05, 0.95)):
    try:
        metric = fn(cell)
    except Exception:
        return {f"{name}": None, f"{name}_sd": None, f"{name}_lower_ci": None, f"{name}_upper_ci": None}


    if np.isscalar(metric) or len(metric) == 1:
        return {f"{name}": metric, f"{name}_sd": 0, f"{name}_lower_ci": None, f"{name}_upper_ci": None}

    point = np.mean(metric)
    lower, upper = np.quantile(metric, probs)
    return {f"{name}": point, f"{name}_sd": metric.std(), f"{name}_lower_ci": lower, f"{name}_upper_ci": upper}


def _compute_font_sizes(n_slices: int) -> dict[str, float | int]:
    return {
        "titleFontSize": BASE_AXIS_TITLE_FONT_SIZE * np.exp(-FONT_SIZE_DECAY_FACTOR * (n_slices - 1)),
        "labelFontSize": BASE_AXIS_LABEL_FONT_SIZE * np.exp(-FONT_SIZE_DECAY_FACTOR * (n_slices - 1)) 
    }

def _currency_symbol(triangle: Triangle) -> str:
    code = triangle.metadata[0].currency
    return get_currency_symbol(code, locale="en_US") or "$"

def _concat_slice_charts(charts: list[alt.Chart], **kwargs) -> alt.Chart:
    if len(charts) == 1:
        return charts[0].properties(**kwargs)

    ncols = max(2, np.ceil(len(charts) / 4))
    fig = alt.concat(*charts, autosize="pad", columns=ncols, **kwargs)
    return fig

