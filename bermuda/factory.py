from __future__ import annotations

import importlib
from functools import update_wrapper

from .triangle import Triangle


def _module_attr(module_name: str, attr_name: str):
    module = importlib.import_module(module_name, __package__)
    return getattr(module, attr_name)


def _bind_method(module_name: str, attr_name: str, triangles_arg: bool = False):
    resolved = None

    def get_func():
        nonlocal resolved
        if resolved is None:
            resolved = _module_attr(module_name, attr_name)
            update_wrapper(method, resolved)
        return resolved

    def method(*args, **kwargs):
        func = get_func()
        if not triangles_arg:
            return func(args[0], *args[1:], **kwargs)
        self, *rest = args
        triangles = kwargs.pop("triangles", None)
        if triangles is None:
            if not rest:
                raise TypeError(f"{attr_name}() missing required argument: 'triangles'")
            triangles, *rest = rest
        return func([self, *triangles], *rest, **kwargs)

    method.__name__ = attr_name
    method.__qualname__ = f"Triangle.{attr_name}"
    return method


def _bind_staticmethod(module_name: str, attr_name: str):
    resolved = None

    def get_func():
        nonlocal resolved
        if resolved is None:
            resolved = _module_attr(module_name, attr_name)
            update_wrapper(method, resolved)
        return resolved

    def method(*args, **kwargs):
        func = get_func()
        return func(*args, **kwargs)

    method.__name__ = attr_name
    method.__qualname__ = f"Triangle.{attr_name}"
    return staticmethod(method)


Triangle.aggregate = _bind_method(".utils.aggregate", "aggregate")
Triangle.summarize = _bind_method(".utils.summarize", "summarize")
Triangle.blend = _bind_method(".utils.summarize", "blend", triangles_arg=True)
Triangle.split = _bind_method(".utils.summarize", "split")
Triangle.merge = _bind_method(".utils.merge", "merge")
Triangle.period_merge = _bind_method(".utils.merge", "period_merge")
Triangle.coalesce = _bind_method(".utils.merge", "coalesce", triangles_arg=True)
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
