from __future__ import annotations

import altair as alt
from babel.numbers import get_currency_symbol
from dataclasses import dataclass
from functools import cache, wraps
from frozendict import frozendict
import json
import pandas as pd
import numpy as np
import string
from typing import Callable, Any, Literal

from .triangle import Triangle, Cell
from .base import metadata_diff

alt.renderers.enable("browser")

SLICE_TITLE_KWARGS = {
    "anchor": "middle",
    "font": "sans-serif",
    "fontWeight": "normal",
    "fontSize": 12,
}

BASE_HEIGHT = 600
BASE_WIDTH = "container"

BASE_AXIS_LABEL_FONT_SIZE = 16
BASE_AXIS_TITLE_FONT_SIZE = 18
FONT_SIZE_DECAY_FACTOR = 0.2

CellArgs = Cell | Cell, Cell, Cell
MetricFunc = Callable[[CellArgs], float | int | np.ndarray]
MetricFuncDict = dict[str, MetricFunc]

COMMON_METRIC_DICT: MetricFuncDict = {
    "Paid Loss Ratio": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"],
    "Reported Loss Ratio": lambda cell: 100
    * cell["reported_loss"]
    / cell["earned_premium"],
    "Incurred Loss Ratio": lambda cell: 100
    * cell["incurred_loss"]
    / cell["earned_premium"],
    "Paid Loss": lambda cell: cell["paid_loss"],
    "Reported Loss": lambda cell: cell["reported_loss"],
    "Incurred Loss": lambda cell: cell["incurred_loss"],
    "Earned Premium": lambda cell: cell["earned_premium"],
    "Reported Claims": lambda cell: cell["reported_claims"],
    "Paid ATA": lambda cell, prev_cell: cell["paid_loss"] / prev_cell["paid_loss"],
    "Reported ATA": lambda cell, prev_cell: cell["reported_loss"]
    / prev_cell["reported_loss"],
    "Paid Incremental ATA": lambda cell, prev_cell: cell["paid_loss"]
    / prev_cell["paid_loss"]
    - 1,
    "Reported Incremental ATA": lambda cell, prev_cell: cell["reported_loss"]
    / prev_cell["reported_loss"]
    - 1,
}

MetricFuncSpec = MetricFuncDict | str | list[str]


@dataclass
class FieldSummary(object):
    field: str
    metric: float | np.ndarray | None
    mean: float | None = None
    median: float | None = None
    sd: float | None = None
    min: float | None = None
    max: float | None = None
    q2_5: float | None = None
    q5: float | None = None
    q10: float | None = None
    q20: float | None = None
    q50: float | None = None
    q80: float | None = None
    q90: float | None = None
    q95: float | None = None
    q97_5: float | None = None
    keep_samples: bool = False

    def __post_init__(self):
        if self.metric is not None:
            if self.keep_samples:
                self.metric = {i: v for i, v in enumerate(self.metric)}
            else:
                self.metric = np.mean(self.metric)
            if self.mean is None:
                self.mean = np.mean(self.metric)

    def tooltip(self, unit: str = "") -> str:
        mean = f"{self.mean:,.{self.precision}f}"
        sd = "" if self.sd is None else f" (SD: {self.sd:,.{self.precision}f})"
        if unit:
            return f"{self.field} ({unit}): {mean}{sd}"
        return f"{self.field}: {mean}{sd}"

    @property
    def precision(self) -> str:
        return "0" if self.mean > 100 else "2"

    @property
    def snake_case_field(self):
        return _to_snake_case(self.field)

    @staticmethod
    def quantiles() -> list[float]:
        return [0.25, 0.05, 0.1, 0.2, 0.5, 0.6, 0.9, 0.95, 0.975]

    @classmethod
    def from_metric(
        cls, name: str, metric: np.ndarray, keep_samples: bool = False
    ) -> FieldSummary:
        return cls(
            name,
            metric,
            np.mean(metric),
            np.median(metric),
            np.std(metric),
            np.min(metric),
            np.max(metric),
            *np.quantile(metric, cls.quantiles()),
            keep_samples=keep_samples,
        )

    def dict(self, return_empty: bool = False, unit: str = ""):
        if self.mean is None:
            return {}
        return {
            **self.__dict__,
            "tooltip": self.tooltip(unit),
            "snake_case_field": self.snake_case_field,
            "unit": unit,
        }

    def json(self, return_empty: bool = False, unit: str = ""):
        return json.dumps(
            {
                k: "null" if v is None else v
                for k, v in self.dict(return_empty, unit).items()
            }
        )


def _resolve_metric_spec(metric_spec: MetricFuncSpec) -> MetricFuncDict:
    if isinstance(metric_spec, str):
        metric_spec = [metric_spec]
    if isinstance(metric_spec, list):
        result = {}
        for ref in metric_spec:
            if not isinstance(ref, str):
                raise ValueError("Supplied metric references must be strings")
            elif ref not in COMMON_METRIC_DICT:
                raise ValueError(f"Don't know the definition of metric {ref}")
            else:
                result[ref] = COMMON_METRIC_DICT[ref]
        return result
    else:
        return metric_spec


# Pattern from https://stackoverflow.com/a/53394430
def freezeargs(func):
    """Convert a mutable dictionary into immutable.
    Useful to be compatible with cache
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = (frozendict(arg) if isinstance(arg, dict) else arg for arg in args)
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def _flatten_dict(x: dict, prepend: str = "") -> dict[str, Any]:
    out = {}
    for key, value in x.items():
        if isinstance(value, dict):
            out |= _flatten_dict(value, prepend=str(key) + "_")
        else:
            out[prepend + str(key)] = value
    return out


def _to_snake_case(x: str) -> str:
    return x.replace(" ", "_").lower()


@freezeargs
@cache
def build_plot_data(
    triangle: Triangle,
    metric_dict: MetricFuncDict | None = None,
    remove_empties: bool = True,
    flat: bool = False,
    keep_samples: bool = False,
) -> list[dict[str, Any]]:
    """Convenience function for building plot data.

    By default, this function will calculate all plot data
    for metrics in COMMON_METRIC_DICT. If bespoke metrics
    are needed, users can pass in a `metric_dict` of their own.

    The returned object is a list of dictionaries. If `flat=False`,
    then each dictionary is nested, with each metric having it's own
    metric spec, e.g. [{'paid_loss_ratio': {'mean': 0.25, 'sd': 0.05}}].
    If `flat=True`, then the dictionaries are non-nested,
    e.g. [{'paid_loss_ratio_mean': 0.25, 'paid_loss_ratio_sd': 0.05}}].

    Args:
        triangle: The triangle to plot.
        metric_dict: A MetricFuncDict to override the default metrics used.
        remove_empties: Remove any cells that don't have data.
        flat: Return a flat dictionary.
        keep_samples: Keep field samples (e.g. distribution variates) in the summary.
    """
    currency = _currency_symbol(triangle)
    unit_lookup = {
        field: (
            currency
            if ("loss" in field.lower() or "premium" in field.lower())
            and "ratio" not in field.lower()
            else "%"
            if "ratio" in field.lower()
            else "%"
            if "share" in field.lower()
            else "N"
            if "claims" in field.lower()
            else ""
        )
        for field in (metric_dict or COMMON_METRIC_DICT)
    }
    field_summaries = {
        cell: {
            _to_snake_case(name): _calculate_field_summary(
                cell, prev_cell, metric, name, keep_samples=keep_samples
            ).dict(remove_empties, unit_lookup.get(name, ""))
            for name, metric in (metric_dict or COMMON_METRIC_DICT).items()
            #{**COMMON_METRIC_DICT, **(metric_dict or {})}.items()
        }
        for cell, prev_cell in zip(triangle, [None, *triangle[:-1]])
    }

    if remove_empties:
        field_summaries = {
            cell: {name: summary for name, summary in values.items() if summary}
            for cell, values in field_summaries.items()
        }

    plot_data = [
        {
            **_core_plot_data(cell),
            "last_lag": max(
                triangle.filter(lambda ob: ob.period == cell.period).dev_lags()
            ),
            "n_fields": len(cell.values),
            "tooltip": ", ".join(
                [
                    v["tooltip"]
                    for v in field_summaries[cell].values()
                    if v["snake_case_field"] in cell.values
                ]
            ),
            **field_summaries[cell],
        }
        for cell in triangle
    ]
    if flat:
        return [_flatten_dict(v) for v in plot_data]
    return plot_data


@alt.theme.register("bermuda_plot_theme", enable=True)
def bermuda_plot_theme() -> alt.theme.ThemeConfig:
    return {
        "autosize": {"contains": "content", "resize": True},
        "config": {
            "style": {
                "group-title": {"fontSize": 24},
                "group-subtitle": {"fontSize": 18},
                "guide-label": {
                    "fontSize": BASE_AXIS_LABEL_FONT_SIZE,
                    "font": "sans-serif",
                },
                "guide-title": {
                    "fontSize": BASE_AXIS_TITLE_FONT_SIZE,
                    "font": "sans-serif",
                },
            },
            "mark": {"color": "black"},
            "title": {"anchor": "start", "offset": 20},
            "axis": {"labelOverlap": True},
            "legend": {
                "orient": "right",
                "titleAnchor": "start",
                "layout": {
                    "direction": "vertical",
                },
            },
        },
    }


def _remove_triangle_samples(triangle: Triangle) -> Triangle:
    """Removes cells that contain samples. The primary use-case
    of this method is to remove future predictions from a triangle to make
    investigating observed data in combined triangles easier."""
    if triangle.num_samples == 1:
        return triangle

    int_cells = []
    for cell in triangle:
        if not any(
            isinstance(v, np.ndarray) and v.size > 1 for v in cell.values.values()
        ):
            int_cells.append(cell)
    return Triangle(int_cells)


def plot_right_edge(
    triangle: Triangle,
    hide_samples: bool = False,
    uncertainty: bool = True,
    uncertainty_type: Literal["ribbon", "segments"] = "ribbon",
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    main_title = alt.Title(
        "Latest Loss Ratio", subtitle="The most recent loss ratio diagonal"
    )
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices)
    fig = _build_metric_slice_charts(
        _remove_triangle_samples(triangle) if hide_samples else triangle,
        plot_func=_plot_right_edge,
        title=main_title,
        facet_titles=facet_titles,
        uncertainty=uncertainty,
        uncertainty_type=uncertainty_type,
        width=width,
        height=height,
        mark_scaler=max_cols,
        ncols=max_cols,
    ).configure_axis(**_compute_font_sizes(max_cols))
    return fig


def _plot_right_edge(
    triangle: Triangle,
    title: alt.Title,
    mark_scaler: int,
    uncertainty: bool = True,
    uncertainty_type: Literal["ribbon", "segments"] = "ribbon",
) -> alt.Chart:
    if "earned_premium" not in triangle.fields:
        raise ValueError(
            "Triangle must contain `earned_premium` to plot its right edge. "
            f"This triangle contains {triangle.fields}"
        )

    data = alt.Data(values=build_plot_data(triangle.right_edge, COMMON_METRIC_DICT))

    currency = _currency_symbol(triangle)

    bar = (
        alt.Chart(data, title=title)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(period_start):O"),
            y=alt.Y(f"earned_premium.mean:Q").axis(
                title="Earned Premium", format="$.2s"
            ),
            color=alt.Color("earned_premium.field:N")
            .scale(range=["lightgray"])
            .title("Field"),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period Start"),
                alt.Tooltip("period_end:T", title="Period End"),
                alt.Tooltip("dev_lag:O", title="Dev Lag"),
                alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
                alt.Tooltip(
                    f"earned_premium.tooltip:N",
                    title="Bar",
                ),
            ],
        )
    )

    loss_error = alt.LayerChart()
    loss_fields = sorted(
        set(
            [
                key
                for cell_values in data.values
                for key in cell_values
                if "loss_ratio" in key
            ]
        )
    )
    lines = alt.LayerChart()
    points = alt.LayerChart()

    for field in loss_fields:
        snake_name = _to_snake_case(field)
        if uncertainty and uncertainty_type == "ribbon":
            loss_error += (
                alt.Chart(data)
                .mark_area(
                    opacity=0.5,
                )
                .encode(
                    x=alt.X("yearmonth(period_start):T"),
                    y=alt.Y(f"{snake_name}.q5:Q").axis(title="Loss Ratio %", format=".0f"),
                    y2=alt.Y2(f"{snake_name}.q95:Q"),
                    color=alt.Color(f"{snake_name}.field:N"),
                )
            )
        elif uncertainty and uncertainty_type == "segments":
            loss_error += (
                alt.Chart(data)
                .mark_rule(thickness=3)
                .encode(
                    x=alt.X("yearmonth(period_start):T").title("Period Start"),
                    y=alt.Y(f"{snake_name}.q5:Q").axis(title="Loss Ratio %", format=".0f"),
                    y2=alt.Y2(f"{snake_name}.q95:Q"),
                    color=alt.Color(f"{snake_name}.field:N"),
                )
            )

        lines += (
            alt.Chart(data)
            .mark_line(
                size=1,
            )
            .encode(
                x=alt.X("yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title(
                    "Period Start"
                ),
                y=alt.Y(
                    f"{snake_name}.mean:Q",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(format=".0f"),
                ).title("Loss Ratio %"),
                color=alt.Color(f"{snake_name}.field:N").legend(title=None),
            )
        ).interactive()

        points += (
            alt.Chart(data)
            .mark_point(
                size=max(20, 100 / mark_scaler),
                filled=True,
                opacity=1,
            )
            .encode(
                x=alt.X("yearmonth(period_start):T", axis=alt.Axis(labelAngle=0)).title(
                    "Period Start"
                ),
                y=alt.Y(
                    f"{snake_name}.mean:Q",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(format=".0f"),
                ).title("Loss Ratio %"),
                color=alt.Color(f"{field}.field:N").legend(title=None),
                tooltip=[
                    alt.Tooltip("period_start:T", title="Period Start"),
                    alt.Tooltip("period_end:T", title="Period End"),
                    alt.Tooltip("dev_lag:O", title="Dev Lag"),
                    alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
                    alt.Tooltip(f"{snake_name}.tooltip:N", title="Field"),
                ],
            )
        )

    fig = alt.layer(bar, loss_error + lines + points).resolve_scale(
        y="independent",
        color="independent",
    )

    return fig.interactive()


def plot_data_completeness(
    triangle: Triangle,
    hide_samples: bool = False,
    width: int = 400,
    height: int = 300,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    main_title = alt.Title(
        "Triangle Completeness",
        subtitle="The number of data fields available per cell",
    )
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices)
    fig = _build_metric_slice_charts(
        _remove_triangle_samples(triangle) if hide_samples else triangle,
        plot_func=_plot_data_completeness,
        title=main_title,
        facet_titles=facet_titles,
        width=width,
        height=height,
        mark_scaler=max_cols,
        ncols=max_cols,
    ).configure_axis(**_compute_font_sizes(max_cols))
    return fig


def _plot_data_completeness(
    triangle: Triangle, title: alt.Title, mark_scaler: int
) -> alt.Chart:
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

    data = alt.Data(values=build_plot_data(triangle))

    fig = (
        alt.Chart(
            data,
            title=title,
        )
        .mark_circle(size=500 * 1 / mark_scaler, opacity=1)
        .encode(
            alt.X(
                "dev_lag:N", axis=alt.Axis(labelAngle=0), scale=alt.Scale(zero=True)
            ).title("Dev Lag (months)"),
            alt.Y(
                "yearmonth(period_start):T", scale=alt.Scale(padding=15, reverse=True)
            ).title("Period Start"),
            color=alt.condition(
                selection,
                alt.Color("n_fields:N").scale(scheme="dark2").title("Number of Fields"),
                alt.value("lightgray"),
            ),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period Start"),
                alt.Tooltip("period_end:T", title="Period End"),
                alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
                alt.Tooltip("dev_lag:N", title="Dev Lag (months)"),
                alt.Tooltip("tooltip:N", title="Fields"),
            ],
        )
        .add_params(selection)
    )

    return fig.interactive()


def plot_heatmap(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = ["Paid Loss Ratio"],
    hide_samples: bool = False,
    show_values: bool = True,
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    """Plot triangle metrics as a heatmap."""
    main_title = alt.Title(
        "Triangle Heatmap",
    )
    metric_dict = _resolve_metric_spec(metric_spec)
    n_metrics = len(metric_dict)
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_heatmap,
            metric_dict=metric_dict,
            title=main_title,
            facet_titles=facet_titles,
            width=width,
            height=height,
            mark_scaler=max_cols,
            ncols=max_cols,
            show_values=show_values,
        )
        .resolve_scale(color="independent")
        .resolve_legend(color="independent")
    )
    return fig


def _plot_heatmap(
    triangle: Triangle,
    metric: MetricFunc,
    name: str,
    show_values: bool,
    title: alt.Title,
    mark_scaler: int,
) -> alt.Chart:
    snake_name = _to_snake_case(name)
    data = alt.Data(values=build_plot_data(triangle, {name: metric}))

    base = alt.Chart(data, title=title).encode(
        x=alt.X("dev_lag:N", axis=alt.Axis(labelAngle=0)).title("Dev Lag (months)"),
        y=alt.X("yearmonth(period_start):O", scale=alt.Scale(reverse=False)).title(
            "Period Start"
        ),
    )

    selection = alt.selection_interval()
    heatmap = (
        base.mark_rect()
        .encode(
            color=alt.when(selection)
            .then(
                alt.Color(
                    f"{snake_name}.mean:Q",
                    scale=alt.Scale(scheme="blueorange"),
                    legend=alt.Legend(title=name, format=".2s"),
                ).title(name)
            )
            .otherwise(
                alt.value("gray"),
            ),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period Start"),
                alt.Tooltip("period_end:T", title="Period End"),
                alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
                alt.Tooltip("dev_lag:O", title="Dev Lag (months)"),
                alt.Tooltip(f"{snake_name}.tooltip:N", title="Field"),
            ],
        )
        .add_params(selection)
    )

    if show_values:
        text = base.mark_text(
            fontSize=BASE_AXIS_TITLE_FONT_SIZE
            * np.exp(-FONT_SIZE_DECAY_FACTOR * mark_scaler),
            font="monospace",
        ).encode(text=alt.Text(f"{snake_name}.mean:Q", format=".0f"))

        return heatmap + text
    return heatmap.resolve_scale(color="independent")


def plot_atas(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = ["Paid ATA"],
    hide_samples: bool = False,
    ncols: int | None = None,
    width: int = 400,
    height: int = 200,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    """Plot triangle ATAs."""
    main_title = alt.Title(
        "Triangle ATAs",
    )
    metric_dict = _resolve_metric_spec(metric_spec)
    n_metrics = len(metric_dict)
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_atas,
            metric_dict=metric_dict,
            title=main_title,
            facet_titles=facet_titles,
            width=width,
            height=height,
            ncols=max_cols,
        )
        .configure_axis(**_compute_font_sizes(max_cols))
        .configure_legend(**_compute_font_sizes(max_cols))
        .resolve_scale(color="independent")
    )
    return fig


def _plot_atas(
    triangle: Triangle, metric: MetricFunc, name: str, title: alt.Title
) -> alt.Chart:
    data = alt.Data(values=build_plot_data(triangle, {name: metric}, flat=True))
    snake_name = _to_snake_case(name)

    tooltip = [
        alt.Tooltip("period_start:T", title="Period Start"),
        alt.Tooltip("period_end:T", title="Period End"),
        alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
        alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
        alt.Tooltip(f"{snake_name}_tooltip:N", title="Field"),
    ]

    base = alt.Chart(data, title=title).encode(
        x=alt.X("dev_lag:Q").title("Dev Lag (months)").scale(padding=10),
        y=alt.Y(f"{snake_name}_mean:Q").title(name).scale(zero=False, padding=10),
        tooltip=tooltip,
    )

    points = base.mark_point(color="black", filled=True)
    boxplot = base.mark_boxplot(
        opacity=0.7,
        color="skyblue",
        median=alt.MarkConfig(stroke="black"),
        rule=alt.MarkConfig(stroke="black"),
        box=alt.MarkConfig(stroke="black"),
    )
    errors = base.mark_rule(thickness=1).encode(
        y=alt.Y(f"{snake_name}_:Q").axis(title=name),
        y2=alt.Y2(f"{snake_name}_q95:Q"),
        color=alt.value("black"),
        opacity=alt.value(0.7),
    )

    return (points + errors + boxplot).interactive()


def plot_growth_curve(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = ["Paid Loss Ratio"],
    hide_samples: bool = False,
    uncertainty: bool = True,
    uncertainty_type: Literal["ribbon", "segments", "spaghetti"] = "ribbon",
    n_lines: int = 100,
    seed: int | None = None,
    width: int = 400,
    height: int = 300,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    """Plot triangle metrics as a growth curve."""
    main_title = alt.Title(
        "Triangle Growth Curve",
    )
    metric_dict = _resolve_metric_spec(metric_spec)
    n_metrics = len(metric_dict)
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_growth_curve,
            metric_dict=metric_dict,
            uncertainty=uncertainty,
            uncertainty_type=uncertainty_type,
            n_lines=n_lines,
            seed=seed,
            mark_scaler=max_cols,
            title=main_title,
            facet_titles=facet_titles,
            width=width,
            height=height,
            ncols=max_cols,
        )
        .configure_axis(**_compute_font_sizes(max_cols))
        .configure_legend(**_compute_font_sizes(max_cols))
    )
    return fig


def _plot_growth_curve(
    triangle: Triangle,
    metric: MetricFunc,
    name: str,
    title: alt.Title,
    uncertainty: bool,
    uncertainty_type: Literal["ribbon", "segments", "spaghetti"],
    mark_scaler: int,
    n_lines: int = 100,
    seed: int | None = None,
) -> alt.Chart:
    snake_name = _to_snake_case(name)

    if uncertainty_type == "spaghetti":
        triangle_thinned = triangle.thin(num_samples=n_lines)
        data = alt.Data(
            values=build_plot_data(
                triangle_thinned, metric_dict={name: metric}, keep_samples=True
            )
        )
    else:
        data = alt.Data(values=build_plot_data(triangle, metric_dict={name: metric}))

    period_start_dtype = "Q" if len(triangle.periods) > 10 else "N"
    color = (
        alt.Color(f"yearmonth(period_start):{period_start_dtype}")
        .scale(scheme="blueorange", reverse=True)
        .legend(title="Period Start")
    )
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["period_start"])
    color_conditional = alt.when(selector).then(color).otherwise(alt.value("lightgray"))
    color_conditional_no_legend = (
        alt.when(selector).then(color_none).otherwise(alt.value("lightgray"))
    )
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )

    base = alt.Chart(data, title=title).encode(
        x=alt.X(
            "dev_lag:Q",
            axis=alt.Axis(grid=True, labelAngle=0),
            scale=alt.Scale(padding=5),
        ).title("Dev Lag (months)"),
        y=alt.Y(f"{snake_name}.mean:Q", axis=alt.Axis(format=".2s")).title(name),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
            alt.Tooltip(f"{snake_name}.tooltip:N", title="Field"),
        ],
    )

    lines = base.mark_line(opacity=0.2).encode(color=color_conditional_no_legend)

    points = base.mark_point(stroke=None, filled=True).encode(
        color=color_conditional_no_legend,
        opacity=opacity_conditional,
    )

    ultimates = (
        base.mark_point(size=300 / mark_scaler, filled=True, stroke=None)
        .encode(
            color=color_conditional,
            opacity=opacity_conditional,
            strokeOpacity=opacity_conditional,
        )
        .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
    )

    if uncertainty and uncertainty_type == "ribbon":
        ribbon_opacity_conditional = (
            alt.when(selector)
            .then(alt.OpacityValue(0.5))
            .otherwise(alt.OpacityValue(0.2))
        )
        errors = base.mark_area(
            opacity=0.5,
        ).encode(
            y=alt.Y(f"{snake_name}.q5:Q"),
            y2=alt.Y2(f"{snake_name}.q95:Q"),
            color=color_conditional_no_legend,
            opacity=ribbon_opacity_conditional,
        )
    elif uncertainty and uncertainty_type == "segments":
        errors = base.mark_rule(thickness=5).encode(
            y=alt.Y(f"{snake_name}.q5:Q").title(name),
            y2=alt.Y2(f"{snake_name}.q95:Q"),
            color=color_conditional_no_legend,
            opacity=opacity_conditional,
        )
    elif uncertainty and uncertainty_type == "spaghetti":
        errors = alt.LayerChart()
        for i in range(n_lines):
            errors += (
                alt.Chart(data)
                .mark_line(opacity=0.2)
                .encode(
                    x=alt.X("dev_lag:Q"),
                    y=alt.Y(f"{snake_name}.metric[{i}]:Q"),
                    detail=f"{i}:N",
                    color=color_conditional_no_legend,
                )
            )
    else:
        errors = alt.LayerChart()

    if len(triangle.periods) == 1:
        scale_color = "shared"
    else:
        scale_color = "independent"

    return (
        alt.layer(errors + lines + points, ultimates.add_params(selector))
        .resolve_scale(color=scale_color)
        .interactive()
    )


def plot_sunset(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = ["Paid Incremental ATA"],
    hide_samples: bool = False,
    uncertainty: bool = True,
    uncertainty_type: Literal["ribbon", "segments"] = "ribbon",
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    """Plot triangle metrics as a sunset."""
    main_title = alt.Title(
        "Triangle Sunset",
    )
    metric_dict = _resolve_metric_spec(metric_spec)
    n_metrics = len(metric_dict)
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_sunset,
            metric_dict=metric_dict,
            uncertainty=uncertainty,
            uncertainty_type=uncertainty_type,
            mark_scaler=max_cols,
            title=main_title,
            facet_titles=facet_titles,
            width=width,
            height=height,
            ncols=max_cols,
        )
        .configure_axis(**_compute_font_sizes(max_cols))
        .configure_legend(**_compute_font_sizes(max_cols))
    )
    return fig.interactive()


def _plot_sunset(
    triangle: Triangle,
    metric: MetricFunc,
    name: str,
    title: alt.Title,
    mark_scaler: int,
    uncertainty: bool,
    uncertainty_type: Literal["ribbon", "segments"],
) -> alt.Chart:
    snake_name = _to_snake_case(name)
    data = alt.Data(values=build_plot_data(triangle, {name: metric}, flat=True))

    color = (
        alt.Color("dev_lag:Q")
        .scale(scheme="blueorange")
        .legend(title="Development Lag")
    )
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["dev_lag"])
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )
    color_conditional = alt.when(selector).then(color).otherwise(alt.value("lightgray"))
    color_conditional_no_legend = (
        alt.when(selector).then(color_none).otherwise(alt.value("lightgray"))
    )

    base = alt.Chart(data, title=title).encode(
        x=alt.X(
            "yearmonth(evaluation_date):O", axis=alt.Axis(grid=True, labelAngle=0)
        ).title("Calendar Year"),
        y=alt.X(f"{snake_name}_mean:Q").title(name).scale(type="pow", exponent=0.3),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
            alt.Tooltip(f"{snake_name}_tooltip:N", title="Field"),
        ],
    )

    points = base.mark_point(stroke=None, size=30 / mark_scaler, filled=True).encode(
        color=color_conditional,
        opacity=opacity_conditional,
        strokeOpacity=opacity_conditional,
    )
    regression = (
        base.transform_loess(
            "evaluation_date", f"{snake_name}_mean", groupby=["dev_lag"], bandwidth=0.6
        )
        .mark_line(strokeWidth=2)
        .encode(color=color_conditional_no_legend, opacity=opacity_conditional)
    )

    if uncertainty and uncertainty_type == "ribbon":
        ribbon_conditional = (
            alt.when(selector)
            .then(alt.OpacityValue(0.5))
            .otherwise(alt.OpacityValue(0.2))
        )
        errors = base.mark_area().encode(
            y=alt.Y(f"{snake_name}_q5:Q").axis(title=name),
            y2=alt.Y2(f"{snake_name}_q95:Q"),
            color=color_conditional_no_legend,
            opacity=ribbon_conditional,
        )
    elif uncertainty and uncertainty_type == "segments":
        errors = base.mark_rule(thickness=5).encode(
            y=alt.Y(f"{snake_name}_q5:Q").axis(title=name),
            y2=alt.Y2(f"{snake_name}_q95:Q"),
            color=color_conditional_no_legend,
            opacity=opacity_conditional,
        )
    else:
        errors = alt.LayerChart()

    return (
        alt.layer(errors, regression, points)
        .add_params(selector)
        .resolve_scale(color="independent")
    )


def plot_mountain(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = ["Paid Loss Ratio"],
    hide_samples: bool = False,
    uncertainty: bool = True,
    uncertainty_type: Literal["ribbon", "segments"] = "ribbon",
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
    highlight_ultimates: bool = True,
) -> alt.Chart:
    """Plot triangle metrics as a mountain."""
    main_title = alt.Title(
        "Triangle Mountain Plot",
    )
    metric_dict = _resolve_metric_spec(metric_spec)
    n_metrics = len(metric_dict)
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    legend_direction = "horizontal" if highlight_ultimates else "vertical"
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_mountain,
            metric_dict=metric_dict,
            uncertainty=uncertainty,
            uncertainty_type=uncertainty_type,
            highlight_ultimates=highlight_ultimates,
            mark_scaler=max_cols,
            title=main_title,
            facet_titles=facet_titles,
            width=width,
            height=height,
            ncols=max_cols,
        )
        .configure_axis(**_compute_font_sizes(max_cols))
        .configure_legend(**_compute_font_sizes(max_cols), direction=legend_direction)
    )
    return fig.interactive()


def _plot_mountain(
    triangle: Triangle,
    metric: MetricFunc,
    name: str,
    title: alt.Title,
    mark_scaler: int,
    uncertainty: bool,
    uncertainty_type: Literal["ribbon", "segments"],
    highlight_ultimates: bool = True,
) -> alt.Chart:
    snake_name = _to_snake_case(name)

    if highlight_ultimates:
        metric_triangle = _remove_triangle_samples(triangle)
        data = alt.Data(values=build_plot_data(metric_triangle, {name: metric}))
        ultimate_data = alt.Data(
            values=build_plot_data(triangle.right_edge, {name: metric})
        )
    else:
        data = alt.Data(values=build_plot_data(triangle, {name: metric}))

    color = (
        alt.Color("dev_lag:Q")
        .scale(scheme="blueorange")
        .legend(title="Development Lag")
    )
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["dev_lag"])
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )

    tooltip = [
        alt.Tooltip("period_start:T", title="Period Start"),
        alt.Tooltip("period_end:T", title="Period End"),
        alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
        alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
        alt.Tooltip(f"{snake_name}.tooltip:N", title="Field"),
    ]

    base = alt.Chart(data, title=title).encode(
        x=alt.X(
            "yearmonth(period_start):O", axis=alt.Axis(grid=True, labelAngle=0)
        ).title("Period Start"),
        y=alt.Y(f"{snake_name}.mean:Q", axis=alt.Axis(format=".2s")).title(name),
        tooltip=tooltip,
    )

    lines = base.mark_line().encode(color=color_none, opacity=opacity_conditional)
    points = base.mark_point(filled=True, stroke=None).encode(
        color=color,
        opacity=opacity_conditional,
    )

    if uncertainty and uncertainty_type == "ribbon":
        ribbon_conditional = (
            alt.when(selector)
            .then(alt.OpacityValue(0.5))
            .otherwise(alt.OpacityValue(0.2))
        )
        errors = base.mark_area().encode(
            y=alt.Y(f"{snake_name}.q5:Q"),
            y2=alt.Y2(f"{snake_name}.q95:Q"),
            color=color_none,
            opacity=ribbon_conditional,
        )
    elif uncertainty and uncertainty_type == "segments":
        errors = base.mark_errorbar(thickness=5).encode(
            y=alt.Y(f"{snake_name}.q5:Q").title(name),
            y2=alt.Y2(f"{snake_name}.q95:Q"),
            color=color_none,
            opacity=opacity_conditional,
        )
    else:
        errors = alt.LayerChart()

    if highlight_ultimates:
        ultimate_color = alt.Color("dev_lag:Q").scale(scheme="greys")
        ultimate_base = alt.Chart(ultimate_data).encode(
            x=alt.X(
                "yearmonth(period_start):O", axis=alt.Axis(grid=True, labelAngle=0)
            ).title("Period Start"),
            y=alt.X(f"{snake_name}.mean:Q").title(name),
            tooltip=tooltip,
            order=alt.value(1),
        )
        ultimates = ultimate_base.mark_line().encode(
            color=ultimate_color.legend(title="Ultimate Lag")
        )
        ultimates += ultimate_base.mark_point(filled=True, size=100).encode(
            color=ultimate_color,
        )

        if uncertainty and uncertainty_type == "ribbon":
            ultimates += ultimate_base.mark_area().encode(
                y=alt.Y(f"{snake_name}.q5:Q"),
                y2=alt.Y2(f"{snake_name}.q95:Q"),
                color=ultimate_color,
                opacity=ribbon_conditional,
            )
        if uncertainty and uncertainty_type == "segments":
            ultimates += ultimate_base.mark_rule(thickness=5).encode(
                y=alt.Y(f"{snake_name}.q5:Q").title(name),
                y2=alt.Y2(f"{snake_name}.q95:Q"),
                color=ultimate_color,
                opacity=opacity_conditional,
            )
    else:
        ultimates = alt.LayerChart()

    return alt.layer(
        lines + errors,
        points.add_params(selector),
        ultimates,
    ).resolve_scale(color="independent")


def plot_ballistic(
    triangle: Triangle,
    axis_metrics: MetricFuncDict = {
        "Paid Loss Ratio": lambda cell: 100
        * cell["paid_loss"]
        / cell["earned_premium"],
        "Reported Loss Ratio": lambda cell: 100
        * cell["reported_loss"]
        / cell["earned_premium"],
    },
    hide_samples: bool = False,
    uncertainty: bool = True,
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
    show_points: bool = True,
) -> alt.Chart:
    """Plot triangle metrics as a ballistic."""
    main_title = alt.Title(
        "Triangle Ballistic Plot",
    )
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices)
    fig = _build_metric_slice_charts(
        _remove_triangle_samples(triangle) if hide_samples else triangle,
        plot_func=_plot_ballistic,
        axis_metrics=axis_metrics,
        title=main_title,
        facet_titles=facet_titles,
        uncertainty=uncertainty,
        width=width,
        height=height,
        mark_scaler=max_cols,
        ncols=max_cols,
        show_points=show_points,
    ).configure_axis(**_compute_font_sizes(max_cols))
    return fig


def _plot_ballistic(
    triangle: Triangle,
    axis_metrics: MetricFuncDict,
    title: alt.Title,
    mark_scaler: int,
    uncertainty: bool,
    show_points: bool,
) -> alt.Chart:
    (name_x, name_y), (func_x, func_y) = zip(*axis_metrics.items())
    snake_name_x = _to_snake_case(name_x)
    snake_name_y = _to_snake_case(name_y)

    data = alt.Data(values=build_plot_data(triangle, axis_metrics))

    color_var = "dev_lag" if show_points else "yearmonth(period_start)"
    color_title = "Development Lags (months)" if show_points else "Period Start"
    if show_points:
        color_dtype = "Q" if len(triangle.dev_lags()) > 10 else "N"
    else:
        color_dtype = "Q" if len(triangle.periods) > 10 else "N"

    color = alt.Color(f"{color_var}:{color_dtype}").legend(title=color_title)
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["period_start"])
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )
    color_conditional = alt.when(selector).then(color).otherwise(alt.value("lightgray"))
    color_conditional_no_legend = (
        alt.when(selector).then(color_none).otherwise(alt.value("lightgray"))
    )

    base = alt.Chart(data, title=title).encode(
        x=alt.X(f"{snake_name_x}.mean:Q").title(name_x).axis(grid=True),
        y=alt.X(f"{snake_name_y}.mean:Q").title(name_y).axis(grid=True),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
            alt.Tooltip(f"{snake_name_x}.tooltip:N", title="Field X"),
            alt.Tooltip(f"{snake_name_y}.tooltip:N", title="Field Y"),
        ],
    )

    diagonal = (
        alt.Chart(data)
        .mark_line(color="black", strokeDash=[5, 5])
        .encode(
            x=f"{snake_name_x}.mean:Q",
            y=f"{snake_name_x}.mean:Q",
        )
    )

    lines = base.mark_line(color="black", strokeWidth=0.5).encode(
        detail="period_start:N", opacity=opacity_conditional
    )
    if show_points:
        points = base.mark_point(
            filled=True, size=100 / mark_scaler, stroke=None
        ).encode(color=color_conditional, opacity=opacity_conditional)
    else:
        points = alt.LayerChart()

    ultimates = (
        base.mark_point(size=200 / mark_scaler, filled=True, stroke=None)
        .encode(color=color_conditional, opacity=opacity_conditional)
        .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
    )

    if uncertainty:
        errors = (
            base.mark_rule(thickness=5)
            .encode(
                y=alt.Y(f"{snake_name_y}.q5:Q").axis(title=name_y),
                y2=alt.Y2(f"{snake_name_y}.q95:Q"),
                color=alt.ColorValue("black"),
            )
            .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
        )
    else:
        errors = alt.LayerChart()

    return (
        alt.layer(diagonal, errors + lines, (points + ultimates).add_params(selector))
        .resolve_scale(color="independent")
        .interactive()
    )


def plot_broom(
    triangle: Triangle,
    axis_metrics: MetricFuncDict = {
        "Paid/Reported Ratio": lambda cell: cell["paid_loss"] / cell["reported_loss"],
        "Paid Loss Ratio": lambda cell: 100
        * cell["paid_loss"]
        / cell["earned_premium"],
    },
    hide_samples: bool = False,
    rule: int | None = 1,
    uncertainty: bool = True,
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
    show_points: bool = True,
) -> alt.Chart:
    """Plot triangle metrics as a broom."""
    main_title = alt.Title(
        "Triangle Broom Plot",
    )
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices)
    fig = _build_metric_slice_charts(
        _remove_triangle_samples(triangle) if hide_samples else triangle,
        plot_func=_plot_broom,
        axis_metrics=axis_metrics,
        title=main_title,
        facet_titles=facet_titles,
        uncertainty=uncertainty,
        rule=rule,
        width=width,
        height=height,
        mark_scaler=max_cols,
        ncols=max_cols,
        show_points=show_points,
    ).configure_axis(**_compute_font_sizes(max_cols))
    return fig.interactive()


def _plot_broom(
    triangle: Triangle,
    axis_metrics: MetricFuncDict,
    title: alt.Title,
    mark_scaler: int,
    uncertainty: bool,
    rule: int | None,
    show_points: bool,
) -> alt.Chart:
    (name_x, name_y), (func_x, func_y) = zip(*axis_metrics.items())
    snake_name_x = _to_snake_case(name_x)
    snake_name_y = _to_snake_case(name_y)

    data = alt.Data(values=build_plot_data(triangle, axis_metrics))

    color_var = "dev_lag" if show_points else "yearmonth(period_start)"
    color_title = "Development Lags (months)" if show_points else "Period Start"
    if show_points:
        color_dtype = "Q" if len(triangle.dev_lags()) > 10 else "N"
    else:
        color_dtype = "Q" if len(triangle.periods) > 10 else "N"

    color = alt.Color(f"{color_var}:{color_dtype}").legend(title=color_title)
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["period_start"])
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )
    color_conditional = alt.when(selector).then(color).otherwise(alt.value("lightgray"))
    color_conditional_no_legend = (
        alt.when(selector).then(color_none).otherwise(alt.value("lightgray"))
    )

    base = alt.Chart(data, title=title).encode(
        x=alt.X(f"{snake_name_x}.mean:Q").scale(padding=10, nice=False).title(name_x),
        y=alt.Y(f"{snake_name_y}.mean:Q").title(name_y),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
            alt.Tooltip(f"{snake_name_x}.tooltip:N", title="Field X"),
            alt.Tooltip(f"{snake_name_y}.tooltip:N", title="Field Y"),
        ],
    )

    wall = (
        alt.Chart().mark_rule(strokeDash=[12, 5], opacity=0.5, strokeWidth=2)
    ).encode()
    if rule is not None:
        wall = wall.encode(x=alt.datum(rule))

    lines = base.mark_line(color="black", strokeWidth=0.5).encode(
        detail="period_start:N", opacity=opacity_conditional
    )
    if show_points:
        points = base.mark_point(
            filled=True, size=100 / mark_scaler, stroke=None
        ).encode(
            color=color_conditional,
            opacity=opacity_conditional,
            strokeOpacity=opacity_conditional,
        )
    else:
        points = alt.LayerChart()

    ultimates = (
        base.mark_point(size=300 / mark_scaler, filled=True, stroke=None)
        .encode(
            color=color_conditional,
            opacity=opacity_conditional,
            strokeOpacity=opacity_conditional,
        )
        .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
    )

    if uncertainty:
        errors = (
            base.mark_rule(thickness=5)
            .encode(
                x=alt.X(f"{snake_name_x}.q5:Q").axis(title=name_x),
                x2=alt.X2(f"{snake_name_x}.q95:Q"),
                color=alt.ColorValue("black"),
            )
            .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
        )
    else:
        errors = alt.LayerChart()

    return alt.layer(
        errors + lines + wall, (points + ultimates).add_params(selector)
    ).resolve_scale(color="independent")


def plot_drip(
    triangle: Triangle,
    axis_metrics: MetricFuncDict = {
        "Reported Loss Ratio": lambda cell: 100
        * cell["reported_loss"]
        / cell["earned_premium"],
        "Open Claim Share": lambda cell: 100
        * cell["open_claims"]
        / cell["reported_claims"],
    },
    hide_samples: bool = False,
    uncertainty: bool = True,
    width: int = 400,
    height: int = 300,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
    show_points: bool = True,
) -> alt.Chart:
    """Plot triangle metrics as a drip."""
    main_title = alt.Title(
        "Triangle Drip Plot",
    )
    n_slices = len(triangle.slices)
    max_cols = ncols or _determine_facet_cols(n_slices)
    fig = (
        _build_metric_slice_charts(
            _remove_triangle_samples(triangle) if hide_samples else triangle,
            plot_func=_plot_drip,
            axis_metrics=axis_metrics,
            title=main_title,
            facet_titles=facet_titles,
            uncertainty=uncertainty,
            width=width,
            height=height,
            mark_scaler=max_cols,
            ncols=max_cols,
            show_points=show_points,
        )
        .configure_axis(**_compute_font_sizes(max_cols))
        .resolve_scale(color="independent")
    )
    return fig.interactive()


def _plot_drip(
    triangle: Triangle,
    axis_metrics: MetricFuncDict,
    title: alt.Title,
    uncertainty: bool,
    mark_scaler: int,
    show_points: bool,
) -> alt.Chart:
    (name_x, name_y), (func_x, func_y) = zip(*axis_metrics.items())
    snake_name_x = _to_snake_case(name_x)
    snake_name_y = _to_snake_case(name_y)

    data = alt.Data(values=build_plot_data(triangle, axis_metrics))

    color_var = "dev_lag" if show_points else "yearmonth(period_start)"
    color_title = "Development Lags (months)" if show_points else "Period Start"
    if show_points:
        color_dtype = "Q" if len(triangle.dev_lags()) > 10 else "N"
    else:
        color_dtype = "Q" if len(triangle.periods) > 10 else "N"

    color = alt.Color(f"{color_var}:{color_dtype}").legend(title=color_title)
    color_none = color.legend(None)

    selector = alt.selection_point(fields=["period_start"])
    opacity_conditional = (
        alt.when(selector).then(alt.OpacityValue(1)).otherwise(alt.OpacityValue(0.2))
    )
    color_conditional = alt.when(selector).then(color).otherwise(alt.value("lightgray"))
    color_conditional_no_legend = (
        alt.when(selector).then(color_none).otherwise(alt.value("lightgray"))
    )

    base = alt.Chart(data, title=title).encode(
        x=alt.X(f"{snake_name_x}.mean:Q").title(name_x, padding=10),
        y=alt.Y(f"{snake_name_y}.mean:Q").title(name_y).scale(nice=False, padding=10),
        tooltip=[
            alt.Tooltip("period_start:T", title="Period Start"),
            alt.Tooltip("period_end:T", title="Period End"),
            alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
            alt.Tooltip("dev_lag:Q", title="Dev Lag (months)"),
            alt.Tooltip(f"{snake_name_x}.tooltip:N", title="Field X"),
            alt.Tooltip(f"{snake_name_y}.tooltip:N", title="Field Y"),
        ],
    )

    lines = base.mark_line(color="black", strokeWidth=0.5).encode(
        detail="period_start:N", opacity=opacity_conditional
    )

    if show_points:
        points = base.mark_point(
            filled=True, size=100 / mark_scaler, stroke=None
        ).encode(color=color_conditional, opacity=opacity_conditional)
    else:
        points = alt.LayerChart()

    ultimates = (
        base.mark_point(size=300 / mark_scaler, filled=True, stroke=None)
        .encode(color=color_conditional, opacity=opacity_conditional)
        .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
    )

    if uncertainty:
        errors = (
            base.mark_rule(thickness=5)
            .encode(
                y=alt.Y(f"{snake_name_y}.q5:Q").title(name_y),
                y2=alt.Y2(f"{snake_name_y}.q95:Q"),
                color=alt.ColorValue("black"),
            )
            .transform_filter(alt.datum.last_lag == alt.datum.dev_lag)
        )
    else:
        errors = alt.LayerChart()

    return alt.layer(
        errors + lines, (points + ultimates).add_params(selector)
    ).resolve_scale(color="independent")


def plot_hose(
    triangle: Triangle,
    axis_metrics: MetricFuncDict = {
        "Paid Loss Ratio": lambda cell: 100
        * cell["paid_loss"]
        / cell["earned_premium"],
        "Incremental Paid Loss Ratio": lambda cell, prev_cell: 100
        * (
            cell["paid_loss"] / cell["earned_premium"]
            - prev_cell["paid_loss"] / prev_cell["earned_premium"]
        ),
    },
    hide_samples: bool = False,
    uncertainty: bool = True,
    width: int = 400,
    height: int = 300,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
    show_points: bool = True,
) -> alt.Chart:
    return plot_drip(
        triangle,
        axis_metrics,
        hide_samples,
        uncertainty,
        width,
        height,
        ncols,
        facet_titles,
        show_points=show_points,
    ).properties(title="Triangle Hose Plot")


def plot_histogram(
    triangle: Triangle,
    metric_spec: MetricFuncSpec = "Paid Loss",
    right_edge: bool = True,
    hide_samples: bool = False,
    width: int = 400,
    height: int = 200,
    ncols: int | None = None,
    facet_titles: list[str] | None = None,
) -> alt.Chart:
    main_title = alt.Title("Triangle Histogram")
    metric_dict = _resolve_metric_spec(metric_spec)
    n_slices = len(triangle.slices)
    n_metrics = len(metric_spec)
    max_cols = ncols or _determine_facet_cols(n_slices * n_metrics)
    fig = _build_metric_slice_charts(
        _remove_triangle_samples(triangle) if hide_samples else triangle,
        plot_func=_plot_histogram,
        metric_dict=metric_dict,
        title=main_title,
        right_edge=right_edge,
        facet_titles=facet_titles,
        width=width,
        height=height,
        ncols=max_cols,
    ).configure_axis(**_compute_font_sizes(max_cols))
    return fig


def _plot_histogram(
    triangle: Triangle,
    metric: MetricFunc,
    name: str,
    right_edge: bool,
    title: alt.Title,
) -> alt.Chart:
    if right_edge:
        triangle = triangle.right_edge

    snake_name = _to_snake_case(name)

    metric_data = alt.Data(
        values=[
            {
                name: value,
                "iteration": i,
            }
            for cell in triangle
            for i, value in enumerate(_scalar_or_array_to_iter(metric(cell)))
        ]
    )
    data = alt.Data(values=build_plot_data(triangle, {name: metric}))

    histogram = (
        alt.Chart(data, title=title)
        .mark_bar()
        .encode(
            x=alt.X(f"{snake_name}.mean:Q", axis=alt.Axis(format=".2s"))
            .bin({"maxbins": 50})
            .title(name),
            y=alt.Y("count()").title("Count"),
        )
    )

    return histogram


def _build_metric_slice_charts(
    triangle, plot_func, title, facet_titles, width, height, ncols, **plot_kwargs
):
    charts = []
    n_slices = len(triangle.slices)
    for i, (_, triangle_slice) in enumerate(triangle.slices.items()):
        if facet_titles is None:
            slice_title = _slice_label(triangle_slice, triangle)
        else:
            slice_title = facet_titles[i]
        if plot_kwargs.get("metric_dict") is not None:
            for name, metric in plot_kwargs["metric_dict"].items():
                metric_title = (
                    (n_slices > 1) * (slice_title + ": ") + name
                    if facet_titles is None
                    else slice_title
                )
                charts.append(
                    plot_func(
                        triangle=triangle_slice,
                        metric=metric,
                        name=name,
                        title=alt.Title(metric_title, **SLICE_TITLE_KWARGS),
                        **{k: v for k, v in plot_kwargs.items() if k != "metric_dict"},
                    ).properties(width=width, height=height)
                )
        else:
            charts.append(
                plot_func(
                    triangle=triangle_slice,
                    title=alt.Title(slice_title, **SLICE_TITLE_KWARGS),
                    **plot_kwargs,
                ).properties(width=width, height=height)
            )
    fig = (
        _concat_charts(charts, title=title, ncols=ncols)
        .configure_axis(**_compute_font_sizes(ncols))
        .configure_legend(**_compute_font_sizes(ncols))
        .configure_mark(color="#1f8fff")
    )
    return fig


def _core_plot_data(cell: Cell) -> dict[str, Any]:
    return {
        "period_start": pd.to_datetime(cell.period_start),
        "period_end": pd.to_datetime(cell.period_end),
        "evaluation_date": pd.to_datetime(cell.evaluation_date),
        "dev_lag": cell.dev_lag(),
    }


def _calculate_field_summary(
    cell: Cell,
    prev_cell: Cell | None,
    func: MetricFunc,
    name: str,
    keep_samples: bool = False,
) -> FieldSummary:
    metric = _safe_apply_metric(cell, prev_cell, func)

    if metric is None or np.isscalar(metric) or len(metric) == 1:
        return FieldSummary(name, metric)

    return FieldSummary.from_metric(name, metric, keep_samples=keep_samples)


def _safe_apply_metric(cell: Cell, prev_cell: Cell | None, func: MetricFunc):
    try:
        if prev_cell.period != cell.period:
            raise IndexError
        return func(cell, prev_cell)
    except Exception:
        try:
            return func(cell)
        except Exception:
            return None


def _compute_font_sizes(mark_scaler: int) -> dict[str, float | int]:
    return {
        "titleFontSize": BASE_AXIS_TITLE_FONT_SIZE
        * np.exp(-FONT_SIZE_DECAY_FACTOR * (mark_scaler - 1)),
        "labelFontSize": BASE_AXIS_LABEL_FONT_SIZE
        * np.exp(-FONT_SIZE_DECAY_FACTOR * (mark_scaler - 1)),
    }


def _currency_symbol(triangle: Triangle) -> str:
    code = triangle.metadata[0].currency
    return get_currency_symbol(code, locale="en_US") or "$"


def _concat_charts(charts: list[alt.Chart], ncols: int, **kwargs) -> alt.Chart:
    if len(charts) == 1:
        return charts[0].properties(**kwargs)

    fig = alt.concat(*charts, columns=ncols, **kwargs)
    return fig


def _determine_facet_cols(n: int):
    """This is a replication of grDevices::n2mfrow in R"""
    return int(min(n, np.ceil(n / np.sqrt(n))))


def _slice_label(slice_tri: Triangle, base_tri: Triangle):
    slice_metadata = metadata_diff(base_tri.common_metadata, slice_tri.common_metadata)

    # Custom elements
    custom_elems = []
    for label, value in {
        **slice_metadata.details,
        **slice_metadata.loss_details,
    }.items():
        custom_elems.append(f"{string.capwords(label)}: {value}")

    # Bare elements
    bare_elems = []
    if slice_metadata.country is not None:
        bare_elems.append(slice_metadata.country)
    if slice_metadata.reinsurance_basis is not None:
        bare_elems.append(slice_metadata.reinsurance_basis)
    if slice_metadata.loss_definition is not None:
        bare_elems.append(slice_metadata.loss_definition)

    # Decorated elements
    decorated_elems = []
    if slice_metadata.per_occurrence_limit is not None:
        decorated_elems.append(f"limit {slice_metadata.per_occurrence_limit}")
    if slice_metadata.risk_basis is not None:
        decorated_elems.append(f"{slice_metadata.risk_basis} Basis")
    if slice_metadata.currency is not None:
        decorated_elems.append(f"in {slice_metadata.currency}")

    custom_label = ", ".join(custom_elems)
    bare_label = " ".join(bare_elems)
    decorated_label = "(" + ", ".join(decorated_elems) + ")"

    label = ""
    if custom_label:
        label += custom_label
    if label and bare_label:
        label += "; "
    if bare_label:
        label += bare_label
    if label and len(decorated_label) > 2:
        label += " "
    if len(decorated_label) > 2:
        label += decorated_label

    return label


def _scalar_or_array_to_iter(x: float | int | list | np.ndarray) -> np.ndarray:
    if np.isscalar(x) or x is None:
        return np.array([x])
    return x
