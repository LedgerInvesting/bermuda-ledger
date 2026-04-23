from __future__ import annotations

import importlib

from .triangle import Triangle


def _module_attr(module_name: str, attr_name: str):
    module = importlib.import_module(module_name, __package__)
    return getattr(module, attr_name)


def _bind_method(module_name: str, attr_name: str, triangle_arg: str = "self"):
    def method(*args, **kwargs):
        func = _module_attr(module_name, attr_name)
        if triangle_arg == "self":
            return func(args[0], *args[1:], **kwargs)
        if triangle_arg == "self_first_in_list":
            self, triangles, *rest = args
            return func([self, *triangles], *rest, **kwargs)
        if triangle_arg == "self_list_only":
            self, triangles, *rest = args
            return func([self, *triangles], *rest, **kwargs)
        raise ValueError(f"Unsupported triangle arg mode: {triangle_arg}")

    method.__name__ = attr_name
    return method


def _bind_staticmethod(module_name: str, attr_name: str):
    def method(*args, **kwargs):
        func = _module_attr(module_name, attr_name)
        return func(*args, **kwargs)

    method.__name__ = attr_name
    return staticmethod(method)


Triangle.aggregate = _bind_method(".utils.aggregate", "aggregate")
Triangle.summarize = _bind_method(".utils.summarize", "summarize")
Triangle.blend = _bind_method(
    ".utils.summarize", "blend", triangle_arg="self_first_in_list"
)
Triangle.split = _bind_method(".utils.summarize", "split")
Triangle.merge = _bind_method(".utils.merge", "merge")
Triangle.period_merge = _bind_method(".utils.merge", "period_merge")
Triangle.coalesce = _bind_method(
    ".utils.merge", "coalesce", triangle_arg="self_list_only"
)
Triangle.to_incremental = _bind_method(".utils.basis", "to_incremental")
Triangle.to_cumulative = _bind_method(".utils.basis", "to_cumulative")
Triangle.add_statics = _bind_method(".utils.fields", "add_statics")
Triangle.make_right_triangle = _bind_method(".utils.extend", "make_right_triangle")
Triangle.make_right_diagonal = _bind_method(".utils.extend", "make_right_diagonal")
Triangle.thin = _bind_method(".utils.thin", "thin")

Triangle.to_array_data_frame = _bind_method(".io.array", "triangle_to_array_data_frame")
Triangle.to_binary = _bind_method(".io.binary_output", "triangle_to_binary")
Triangle.to_json = _bind_method(".io.json", "triangle_to_json")
Triangle.to_chain_ladder = _bind_method(".io.chain_ladder", "triangle_to_chain_ladder")
Triangle.to_dict = _bind_method(".io.json", "triangle_to_dict")
Triangle.to_long_csv = _bind_method(".io.data_frame_output", "triangle_to_long_csv")
Triangle.to_long_data_frame = _bind_method(
    ".io.data_frame_output", "triangle_to_long_data_frame"
)
Triangle.to_right_edge_data_frame = _bind_method(
    ".io.array", "triangle_to_right_edge_data_frame"
)
Triangle.to_wide_csv = _bind_method(".io.data_frame_output", "triangle_to_wide_csv")
Triangle.to_wide_data_frame = _bind_method(
    ".io.data_frame_output", "triangle_to_wide_data_frame"
)

Triangle.from_array_data_frame = _bind_staticmethod(
    ".io.array", "array_data_frame_to_triangle"
)
Triangle.from_binary = _bind_staticmethod(".io.binary_input", "binary_to_triangle")
Triangle.from_chain_ladder = _bind_staticmethod(
    ".io.chain_ladder", "chain_ladder_to_triangle"
)
Triangle.from_dict = _bind_staticmethod(".io.json", "dict_to_triangle")
Triangle.from_long_csv = _bind_staticmethod(
    ".io.data_frame_input", "long_csv_to_triangle"
)
Triangle.from_long_data_frame = _bind_staticmethod(
    ".io.data_frame_input", "long_data_frame_to_triangle"
)
Triangle.from_statics_data_frame = _bind_staticmethod(
    ".io.array", "statics_data_frame_to_triangle"
)
Triangle.from_wide_csv = _bind_staticmethod(
    ".io.data_frame_input", "wide_csv_to_triangle"
)
Triangle.from_wide_data_frame = _bind_staticmethod(
    ".io.data_frame_input", "wide_data_frame_to_triangle"
)
Triangle.from_json = _bind_staticmethod(".io.json", "json_to_triangle")

Triangle.plot_right_edge = _bind_method(".plot", "plot_right_edge")
Triangle.plot_data_completeness = _bind_method(".plot", "plot_data_completeness")
Triangle.plot_heatmap = _bind_method(".plot", "plot_heatmap")
Triangle.plot_atas = _bind_method(".plot", "plot_atas")
Triangle.plot_growth_curve = _bind_method(".plot", "plot_growth_curve")
Triangle.plot_mountain = _bind_method(".plot", "plot_mountain")
Triangle.plot_ballistic = _bind_method(".plot", "plot_ballistic")
Triangle.plot_broom = _bind_method(".plot", "plot_broom")
Triangle.plot_drip = _bind_method(".plot", "plot_drip")
Triangle.plot_hose = _bind_method(".plot", "plot_hose")
Triangle.plot_sunset = _bind_method(".plot", "plot_sunset")
Triangle.plot_histogram = _bind_method(".plot", "plot_histogram")
