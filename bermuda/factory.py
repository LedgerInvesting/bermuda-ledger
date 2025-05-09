from functools import wraps

from .io import (
    array_data_frame_to_triangle,
    binary_to_triangle,
    chain_ladder_to_triangle,
    dict_to_triangle,
    json_to_triangle,
    long_csv_to_triangle,
    long_data_frame_to_triangle,
    statics_data_frame_to_triangle,
    triangle_to_array_data_frame,
    triangle_to_binary,
    triangle_to_chain_ladder,
    triangle_to_dict,
    triangle_to_json,
    triangle_to_long_csv,
    triangle_to_long_data_frame,
    triangle_to_right_edge_data_frame,
    triangle_to_wide_csv,
    triangle_to_wide_data_frame,
    wide_csv_to_triangle,
    wide_data_frame_to_triangle,
)
from .triangle import Triangle
from .utils import (
    add_statics,
    aggregate,
    blend,
    coalesce,
    make_right_diagonal,
    make_right_triangle,
    merge,
    period_merge,
    split,
    summarize,
    thin,
    to_cumulative,
    to_incremental,
)
from .plot import (
    plot_data_completeness,
    plot_right_edge,
    plot_heatmap,
    plot_growth_curve,
    plot_mountain,
    plot_sunset,
    plot_ballistic,
    plot_broom,
    plot_drip,
    plot_hose,
)

# utils
Triangle.aggregate = wraps(aggregate)(
    lambda self, *args, **kwargs: aggregate(self, *args, **kwargs)
)
Triangle.summarize = wraps(summarize)(
    lambda self, *args, **kwargs: summarize(self, *args, **kwargs)
)
Triangle.blend = wraps(blend)(
    lambda self, triangles, **kwargs: blend(triangles=[self, *triangles], **kwargs)
)
Triangle.split = wraps(split)(
    lambda self, *args, **kwargs: split(self, *args, **kwargs)
)
Triangle.merge = wraps(merge)(
    lambda self, *args, **kwargs: merge(self, *args, **kwargs)
)
Triangle.period_merge = wraps(period_merge)(
    lambda self, *args, **kwargs: period_merge(self, *args, **kwargs)
)
Triangle.coalesce = wraps(coalesce)(
    lambda self, triangles: coalesce([self, *triangles])
)
Triangle.to_incremental = wraps(to_incremental)(lambda self: to_incremental(self))
Triangle.to_cumulative = wraps(to_cumulative)(lambda self: to_cumulative(self))
Triangle.add_statics = wraps(add_statics)(
    lambda self, *args, **kwargs: add_statics(self, *args, **kwargs)
)
Triangle.make_right_triangle = wraps(make_right_triangle)(
    lambda self, *args, **kwargs: make_right_triangle(self, *args, **kwargs)
)
Triangle.make_right_diagonal = wraps(make_right_diagonal)(
    lambda self, *args, **kwargs: make_right_diagonal(self, *args, **kwargs)
)
Triangle.thin = wraps(thin)(lambda self, *args, **kwargs: thin(self, *args, **kwargs))

# io
Triangle.to_array_data_frame = wraps(triangle_to_array_data_frame)(
    lambda self, *args, **kwargs: triangle_to_array_data_frame(self, *args, **kwargs)
)
Triangle.to_binary = wraps(triangle_to_binary)(
    lambda self, *args, **kwargs: triangle_to_binary(self, *args, **kwargs)
)
Triangle.to_json = wraps(triangle_to_json)(triangle_to_json)
Triangle.to_chain_ladder = wraps(triangle_to_chain_ladder)(
    lambda self, *args, **kwargs: triangle_to_chain_ladder(self, *args, **kwargs)
)
Triangle.to_dict = wraps(triangle_to_dict)(triangle_to_dict)
Triangle.to_long_csv = wraps(triangle_to_long_csv)(
    lambda self, *args, **kwargs: triangle_to_long_csv(self, *args, **kwargs)
)
Triangle.to_long_data_frame = wraps(triangle_to_long_data_frame)(
    lambda self, *args, **kwargs: triangle_to_long_data_frame(self, *args, **kwargs)
)
Triangle.to_right_edge_data_frame = wraps(triangle_to_right_edge_data_frame)(
    lambda self, *args, **kwargs: triangle_to_right_edge_data_frame(
        self, *args, **kwargs
    )
)
Triangle.to_wide_csv = wraps(triangle_to_wide_csv)(
    lambda self, *args, **kwargs: triangle_to_wide_csv(self, *args, **kwargs)
)
Triangle.to_wide_data_frame = wraps(triangle_to_wide_data_frame)(
    lambda self, *args, **kwargs: triangle_to_wide_data_frame(self, *args, **kwargs)
)

# constructors
Triangle.from_array_data_frame = staticmethod(
    wraps(array_data_frame_to_triangle)(array_data_frame_to_triangle)
)
Triangle.from_binary = staticmethod(wraps(binary_to_triangle)(binary_to_triangle))
Triangle.from_chain_ladder = staticmethod(
    wraps(chain_ladder_to_triangle)(chain_ladder_to_triangle)
)
Triangle.from_dict = staticmethod(wraps(dict_to_triangle)(dict_to_triangle))
Triangle.from_long_csv = staticmethod(wraps(long_csv_to_triangle)(long_csv_to_triangle))
Triangle.from_long_data_frame = staticmethod(
    wraps(long_data_frame_to_triangle)(long_data_frame_to_triangle)
)
Triangle.from_statics_data_frame = staticmethod(
    wraps(statics_data_frame_to_triangle)(statics_data_frame_to_triangle)
)
Triangle.from_wide_csv = staticmethod(wraps(wide_csv_to_triangle)(wide_csv_to_triangle))
Triangle.from_wide_data_frame = staticmethod(
    wraps(wide_data_frame_to_triangle)(wide_data_frame_to_triangle)
)
Triangle.from_json = staticmethod(wraps(json_to_triangle)(json_to_triangle))

# plots
Triangle.plot_right_edge = wraps(plot_right_edge)(lambda self, *args, **kwargs: plot_right_edge(self, *args, **kwargs))
Triangle.plot_data_completeness = wraps(plot_data_completeness)(lambda self, *args, **kwargs: plot_data_completeness(self, *args, **kwargs))
Triangle.plot_heatmap = wraps(plot_heatmap)(lambda self, *args, **kwargs: plot_heatmap(self, *args, **kwargs))
Triangle.plot_growth_curve = wraps(plot_growth_curve)(lambda self, *args, **kwargs: plot_growth_curve(self, *args, **kwargs))
Triangle.plot_mountain = wraps(plot_mountain)(lambda self, *args, **kwargs: plot_mountain(self, *args, **kwargs))
Triangle.plot_ballistic = wraps(plot_ballistic)(lambda self, *args, **kwargs: plot_ballistic(self, *args, **kwargs))
Triangle.plot_broom = wraps(plot_broom)(lambda self, *args, **kwargs: plot_broom(self, *args, **kwargs))
Triangle.plot_drip = wraps(plot_drip)(lambda self, *args, **kwargs: plot_drip(self, *args, **kwargs))
Triangle.plot_hose = wraps(plot_hose)(lambda self, *args, **kwargs: plot_hose(self, *args, **kwargs))
Triangle.plot_sunset = wraps(plot_sunset)(lambda self, *args, **kwargs: plot_sunset(self, *args, **kwargs))
