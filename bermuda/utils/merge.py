from collections import defaultdict
from warnings import warn

from ..base import Cell
from ..triangle import Triangle
from .join import join

__all__ = [
    "merge",
    "period_merge",
    "loose_period_merge",
    "coalesce",
]


def merge(
    tri1: Triangle,
    tri2: Triangle,
    join_type: str = "full",
    on: list[str] | None = None,
) -> Triangle:
    """Merge cells of two triangles, if coordinates overlap then prioritize right cell values.

    All cells from tri1 and tri2 are preserved by this merge. If two cells overlap by index,
    (period_start, period_end, evaluation_date, metadata), then values from tri2 take
    precedence.

    Merge is a higher-level operation that returns a Triangle of merged cells. For a
    lower-level implementation that returns a list of pairs of matched cells, see `join`.

    `merge` is different from `coalesce` in that it combines the union of fields in
    cells at the same coordinates, where `coalesce` only keeps the fields and values from
    the cell with precedence.

    Args:
        tri1: The lower-precedence, left triangle to merge.
        tri2: The higher-precedence, right triangle to merge.
        join_type: The type of relational join to perform. For a list of all join types
            and their descriptions, see the documentation for `join`.
        on: which metadata attributes and detail keys to join on when merging cell values.
            Default behavior is to consider all metadata attributes when merging cells.
    Returns:
        A triangle with the merged cells from the two triangles.
    """
    cell_pairs = join(tri1, tri2, join_type, on)
    return Triangle([_merge_cell_pair(*cell_pair) for cell_pair in cell_pairs])


def _merge_cell_pair(cell1: Cell | None, cell2: Cell | None) -> Cell:
    if cell1 is None:
        return cell2
    elif cell2 is None:
        return cell1
    else:
        # We can safely assume that cell1 and cell2 cannot both be None
        return cell1.replace(values={**cell1.values, **cell2.values})


def period_merge(
    tri1: Triangle[Cell],
    tri2: Triangle[Cell],
    suffix: str | None = None,
) -> Triangle[Cell]:
    """Add period-level values of tri2 to tri1 cells. tri2 must have a single cell per period index.

    All cells from tri1 are preserved by this merge. If there is a matching cell in tri2 by index
    (period_start, period_end, metadata), then values from tri2 take precedence. There must be
    only one cell in tri2 per index.

    Args:
        tri1: First triangle in the merge.
        tri2: Second triangle in the merge.
        suffix: If present, then every field in tri2 will have the suffix appended to the name
            of the field after the merge to avoid name conflicts. If None, then no suffix
            will be appended and the values in tri2 will overwrite the values in tri1.
    """
    # Make sure that both triangles have the same cell type
    if min(len(tri1), len(tri2)) > 0 and type(tri1.cells[0]) != type(tri2.cells[0]):  # noqa: E721
        raise ValueError("Triangles must have the same cell type")

    tri1_cells = defaultdict(list)
    for cell in tri1:
        tri1_cells[(cell.period_start, cell.period_end, cell.metadata)] += [cell]

    tri2_cells = defaultdict(list)
    for cell in tri2:
        tri2_cells[(cell.period_start, cell.period_end, cell.metadata)] += [cell]

    output_cells = []
    for idx, cells in tri1_cells.items():
        right_cell = tri2_cells[idx]
        if len(right_cell) > 1:
            raise (ValueError(f"multiple cells at tri2 index {idx}"))
        elif len(right_cell) == 0:
            output_cells += cells
        else:
            output_cells += [
                _overwrite_values(cell, right_cell[0], suffix=suffix) for cell in cells
            ]
    return Triangle(output_cells)


def _overwrite_values(cell1: Cell, cell2: Cell, suffix: str | None = None) -> Cell:
    """Replace cell1 values with cell2 values for common fields."""
    if suffix:
        replace_map = {f"{k}{suffix}": v for k, v in cell2.values.items()}
    else:
        replace_map = cell2.values

    return cell1.replace(values={**cell1.values, **replace_map})


def loose_period_merge(
    tri1: Triangle[Cell],
    tri2: Triangle[Cell],
    suffix: str | None = None,
) -> Triangle[Cell]:
    """Add period-level values of tri2 to tri1 cells. tri2 must have a single cell per period index.

    All cells from tri1 are preserved by this merge. If there is a matching cell in tri2 by index
    (period_start, period_end, metadata), then values from tri2 take precedence. There must be
    only one cell in tri2 per index.

    Note that this is different from `period_merge` in that it will allow partial matches on
    metadata details. For examples, if tri1 has a cell with metadata details `{"a": 1, "b": 2}`
    and tri2 has a cell with metadata details `{"a": 1}`, then the cell from tri2 will be merged
    with the cell from tri1. This is useful for merging triangles with different levels of
    granularity, particularly in BFing. Tri1 must be more granular (or as granular) as tri2.

    Args:
        tri1: First triangle in the merge.
        tri2: Second triangle in the merge.
        suffix: If present, then every field in tri2 will have the suffix appended to the name
            of the field after the merge to avoid name conflicts. If None, then no suffix
            will be appended and the values in tri2 will overwrite the values in tri1.
    """
    # Make sure that both triangles have the same cell type
    if min(len(tri1), len(tri2)) > 0 and not isinstance(
        tri1.cells[0], type(tri2.cells[0])
    ):
        raise ValueError("Triangles must have the same cell type")
    if tri1.cells[0].metadata.details.keys() < tri2.cells[0].metadata.details.keys():
        raise ValueError("tri1 must have at least as many metadata details as tri2")

    common_details = tri1.metadata[0].details.keys() & tri2.metadata[0].details.keys()
    tri1_cells = defaultdict(list)
    for cell in tri1:
        metadata_proxy = cell.derive_metadata(
            details=lambda meta: {k: meta.details[k] for k in common_details}
        )
        tri1_cells[(cell.period_start, cell.period_end, metadata_proxy.metadata)] += [
            cell
        ]

    tri2_cells = defaultdict(list)
    for cell in tri2:
        tri2_cells[(cell.period_start, cell.period_end, cell.metadata)] += [cell]

    output_cells = []
    for idx, cells in tri1_cells.items():
        right_cell = tri2_cells[idx]
        if len(right_cell) > 1:
            raise (ValueError(f"multiple cells at tri2 index {idx}"))
        elif len(right_cell) == 0:
            output_cells += cells
        else:
            output_cells += [
                _overwrite_values(cell, right_cell[0], suffix=suffix) for cell in cells
            ]
    return Triangle(output_cells)


def coalesce(triangles: list[Triangle]) -> Triangle:
    """Coalesce triangle cells from a list of Triangles such that the resulting
    triangle is a union of the cells from each triangle. Matching cells in the
    earlier triangles in the triangle list take precedence over cells from later
    triangles.

    `coalesce` is different from `merge` in that it only keeps the fields and values from
    the cell with precedence, wheras `merge` will return the union of fields across the
    matching cells.

    Args:
        triangles: A list of Triangle whose cells will be coalesced.

    Returns:
        A Triangle with Cells at all coordinates of the triangles in the triangle
        list.
    """
    if not isinstance(triangles, list):
        raise ValueError("coalesce must be provided a list of triangles")
    grouped_cells = defaultdict(list)
    for triangle in triangles:
        for cell in triangle:
            grouped_cells[(cell.metadata, cell.period, cell.evaluation_date)].append(
                cell
            )

    if not any(len(matching_cells) > 1 for matching_cells in grouped_cells.values()):
        warn(
            """
        No overlapping cells in the list of triangles provided, check for differences
        in metadata
        """
        )
    return Triangle([cell_options[0] for cell_options in grouped_cells.values()])
