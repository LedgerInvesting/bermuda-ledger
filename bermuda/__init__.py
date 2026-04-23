from __future__ import annotations

import importlib
import os
from typing import Any

_SUBMODULES = {
    "base": ".base",
    "date_utils": ".date_utils",
    "errors": ".errors",
    "factory": ".factory",
    "io": ".io",
    "matrix": ".matrix",
    "plot": ".plot",
    "triangle": ".triangle",
    "utils": ".utils",
}

local_dir = os.path.dirname(os.path.abspath(__file__))

_EXPORTS = {
    "__version__": (".__about__", "__version__"),
    "Cell": (".base", "Cell"),
    "CumulativeCell": (".base", "CumulativeCell"),
    "IncrementalCell": (".base", "IncrementalCell"),
    "Metadata": (".base", "Metadata"),
    "TriangleError": (".errors", "TriangleError"),
    "TriangleEmptyError": (".errors", "TriangleEmptyError"),
    "TriangleWarning": (".errors", "TriangleWarning"),
    "DuplicateCellWarning": (".errors", "DuplicateCellWarning"),
    "array_data_frame_to_triangle": (".io.array", "array_data_frame_to_triangle"),
    "array_triangle_builder": (".io.array", "array_triangle_builder"),
    "statics_data_frame_to_triangle": (".io.array", "statics_data_frame_to_triangle"),
    "triangle_to_array_data_frame": (".io.array", "triangle_to_array_data_frame"),
    "triangle_to_right_edge_data_frame": (
        ".io.array",
        "triangle_to_right_edge_data_frame",
    ),
    "binary_to_triangle": (".io.binary_input", "binary_to_triangle"),
    "triangle_to_binary": (".io.binary_output", "triangle_to_binary"),
    "chain_ladder_to_triangle": (".io.chain_ladder", "chain_ladder_to_triangle"),
    "triangle_to_chain_ladder": (".io.chain_ladder", "triangle_to_chain_ladder"),
    "long_csv_to_triangle": (".io.data_frame_input", "long_csv_to_triangle"),
    "long_data_frame_to_triangle": (
        ".io.data_frame_input",
        "long_data_frame_to_triangle",
    ),
    "wide_csv_to_triangle": (".io.data_frame_input", "wide_csv_to_triangle"),
    "wide_data_frame_to_triangle": (
        ".io.data_frame_input",
        "wide_data_frame_to_triangle",
    ),
    "triangle_to_long_csv": (".io.data_frame_output", "triangle_to_long_csv"),
    "triangle_to_long_data_frame": (
        ".io.data_frame_output",
        "triangle_to_long_data_frame",
    ),
    "triangle_to_wide_csv": (".io.data_frame_output", "triangle_to_wide_csv"),
    "triangle_to_wide_data_frame": (
        ".io.data_frame_output",
        "triangle_to_wide_data_frame",
    ),
    "dict_to_triangle": (".io.json", "dict_to_triangle"),
    "json_string_to_triangle": (".io.json", "json_string_to_triangle"),
    "json_to_triangle": (".io.json", "json_to_triangle"),
    "triangle_json_load": (".io.json", "triangle_json_load"),
    "triangle_json_loads": (".io.json", "triangle_json_loads"),
    "triangle_to_dict": (".io.json", "triangle_to_dict"),
    "triangle_to_json": (".io.json", "triangle_to_json"),
    "triangle_to_matrix": (".io.matrix", "triangle_to_matrix"),
    "matrix_to_triangle": (".io.matrix", "matrix_to_triangle"),
    "triangle_to_rich_matrix": (".io.rich_matrix", "triangle_to_rich_matrix"),
    "rich_matrix_to_triangle": (".io.rich_matrix", "rich_matrix_to_triangle"),
    "DisaggregatedValue": (".matrix", "DisaggregatedValue"),
    "Matrix": (".matrix", "Matrix"),
    "MatrixIndex": (".matrix", "MatrixIndex"),
    "MissingValue": (".matrix", "MissingValue"),
    "PredictedValue": (".matrix", "PredictedValue"),
    "RichMatrix": (".matrix", "RichMatrix"),
    "paid_bs_adjustment": (".utils.adjust", "paid_bs_adjustment"),
    "reported_bs_adjustment": (".utils.adjust", "reported_bs_adjustment"),
    "weight_geometric_decay": (".utils.adjust", "weight_geometric_decay"),
    "aggregate": (".utils.aggregate", "aggregate"),
    "backfill": (".utils.backfill", "backfill"),
    "accident_quarter_to_policy_year": (
        ".utils.basis",
        "accident_quarter_to_policy_year",
    ),
    "to_cumulative": (".utils.basis", "to_cumulative"),
    "to_incremental": (".utils.basis", "to_incremental"),
    "bootstrap": (".utils.bootstrap", "bootstrap"),
    "convert_to_dollars": (".utils.currency", "convert_to_dollars"),
    "convert_currency": (".utils.currency", "convert_currency"),
    "disaggregate": (".utils.disaggregate", "disaggregate"),
    "disaggregate_development": (
        ".utils.disaggregate",
        "disaggregate_development",
    ),
    "disaggregate_experience": (".utils.disaggregate", "disaggregate_experience"),
    "make_right_triangle": (".utils.extend", "make_right_triangle"),
    "make_right_diagonal": (".utils.extend", "make_right_diagonal"),
    "make_pred_triangle": (".utils.extend", "make_pred_triangle"),
    "make_pred_triangle_complement": (
        ".utils.extend",
        "make_pred_triangle_complement",
    ),
    "make_pred_triangle_with_init": (".utils.extend", "make_pred_triangle_with_init"),
    "add_statics": (".utils.fields", "add_statics"),
    "array_from_field": (".utils.fields", "array_from_field"),
    "array_sizes": (".utils.fields", "array_sizes"),
    "array_size": (".utils.fields", "array_size"),
    "fill_forward_gaps": (".utils.fill", "fill_forward_gaps"),
    "join": (".utils.join", "join"),
    "merge": (".utils.merge", "merge"),
    "period_merge": (".utils.merge", "period_merge"),
    "loose_period_merge": (".utils.merge", "loose_period_merge"),
    "coalesce": (".utils.merge", "coalesce"),
    "moment_match": (".utils.method_moments", "moment_match"),
    "program_earned_premium": (".utils.premium_pattern", "program_earned_premium"),
    "shift_origin": (".utils.shift_origin", "shift_origin"),
    "slice_to_triangle": (".utils.slice", "slice_to_triangle"),
    "triangle_to_slice": (".utils.slice", "triangle_to_slice"),
    "summarize_cell_values": (".utils.summarize", "summarize_cell_values"),
    "blend_samples": (".utils.summarize", "blend_samples"),
    "blend_cells": (".utils.summarize", "blend_cells"),
    "blend": (".utils.summarize", "blend"),
    "split": (".utils.summarize", "split"),
    "summarize": (".utils.summarize", "summarize"),
    "thin": (".utils.thin", "thin"),
    "Triangle": (".factory", "Triangle"),
    "TriangleSlice": (".triangle", "TriangleSlice"),
    "DevLag": (".date_utils", "DevLag"),
    "calculate_dev_lag": (".date_utils", "calculate_dev_lag"),
    "add_months": (".date_utils", "add_months"),
    "dev_lag_months": (".date_utils", "dev_lag_months"),
    "standardize_resolution": (".date_utils", "standardize_resolution"),
    "resolution_delta": (".date_utils", "resolution_delta"),
    "month_to_id": (".date_utils", "month_to_id"),
    "id_to_month": (".date_utils", "id_to_month"),
    "period_resolution": (".date_utils", "period_resolution"),
    "eval_date_resolution": (".date_utils", "eval_date_resolution"),
    "is_triangle_monthly": (".date_utils", "is_triangle_monthly"),
    "drop_off_diagonals": (".date_utils", "drop_off_diagonals"),
    "meyers_tri": (".factory", "Triangle"),
}

__all__ = ["os", "local_dir", *_SUBMODULES, *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(_SUBMODULES[name], __name__)
        globals()[name] = module
        return module

    if name == "meyers_tri":
        triangle_cls = getattr(importlib.import_module(".factory", __name__), "Triangle")
        value = triangle_cls.from_binary(os.path.join(local_dir, "meyers.trib"))
        globals()[name] = value
        return value

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
