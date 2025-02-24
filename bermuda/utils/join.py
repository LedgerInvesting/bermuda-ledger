from typing import Tuple

import toolz as tlz

from ..base import Cell, Metadata
from ..triangle import Triangle

__all__ = [
    "join",
]


def join(
    tri1: Triangle,
    tri2: Triangle,
    join_type: str = "full",
    on: list[str] | None = None,
) -> list[Tuple[Cell | None, Cell | None]]:
    """Create pairs of cells that share the same coordinates and metadata from two triangles.

    Join is a lower-level operation that returns a list of pairs of cells. Many common use cases
    are better served by its higher-order cousin, `merge`.

    Args:
        tri1: The left triangle in the join.
        tri2: The right triangle in the join.
        join_type: The type of relational join to perform. Available join types are:
            * inner: Return only cell coordinates that are present in both triangles.
            * left: Return all cell coordinates in the left triangle, along with any
                available matches on the right triangle.
            * right: Return all cell coordinates in the right triangle, along with any
                available matches on the left triangle.
            * full: Return all cell coordinates in both triangles, regardless of
                whether there is a match on the other side.
            * left_anti: Return only those cells in the left triangle that do not have
                matching coordinates in the right triangle.
            * right_anti: Return only those cells in the right triangle that do not
                have matching coordinates in the left triangle.
        on: which metadata attributes and detail keys to join on when merging cell values.
            Default behavior is to consider all metadata attributes when merging cells.
    Returns:
        A list of 2-tuples of `Cell`s. Each tuple represents a pair of cells with the
        same coordinates and metadata. The first and second elements of the tuple are
        the cell from the left and right triangles, respectively. Depending on the
        type of join, either of the cells in the tuple may be `None`, but by construction,
        every tuple will have at least one non-`None` cell.
    """
    # Make sure that both triangles have the same cell type
    if min(len(tri1), len(tri2)) > 0 and type(tri1.cells[0]) != type(tri2.cells[0]):  # noqa: E721
        raise ValueError("Triangles must have the same cell type")

    # If attributes are passed to `on` we will ignore any other metadata by removing all other
    # metadata attributes
    if on:
        tri1 = _select_metadata(tri1, on)
        tri2 = _select_metadata(tri2, on)

    # Create an index of all cells in each triangle
    if tri1.is_incremental:
        tri1_cells = {
            (
                cell.metadata,
                cell.period_start,
                cell.period_end,
                cell.evaluation_date,
                cell.prev_evaluation_date,
            ): cell
            for cell in tri1
        }
        tri2_cells = {
            (
                cell.metadata,
                cell.period_start,
                cell.period_end,
                cell.evaluation_date,
                cell.prev_evaluation_date,
            ): cell
            for cell in tri2
        }
    else:
        tri1_cells = {
            (
                cell.metadata,
                cell.period_start,
                cell.period_end,
                cell.evaluation_date,
            ): cell
            for cell in tri1
        }
        tri2_cells = {
            (
                cell.metadata,
                cell.period_start,
                cell.period_end,
                cell.evaluation_date,
            ): cell
            for cell in tri2
        }

    # Merge them into a single list of 2-tuples
    all_coordinates = set(tri1_cells.keys()) | set(tri2_cells.keys())
    cell_pairs = [
        (tri1_cells.get(coord), tri2_cells.get(coord)) for coord in all_coordinates
    ]

    if join_type == "full":
        return cell_pairs
    elif join_type == "left":
        return [(cell1, cell2) for cell1, cell2 in cell_pairs if cell1 is not None]
    elif join_type == "right":
        return [(cell1, cell2) for cell1, cell2 in cell_pairs if cell2 is not None]
    elif join_type == "inner":
        return [
            (cell1, cell2)
            for cell1, cell2 in cell_pairs
            if cell1 is not None and cell2 is not None
        ]
    elif join_type == "left_anti":
        return [(cell1, cell2) for cell1, cell2 in cell_pairs if cell2 is None]
    elif join_type == "right_anti":
        return [(cell1, cell2) for cell1, cell2 in cell_pairs if cell1 is None]
    else:
        raise ValueError(f"Unrecognized join type `{join_type}`")


def _select_metadata(triangle, attributes_list):
    def metadata_mod(cell):
        new_details = tlz.keyfilter(
            lambda key: key in attributes_list, cell.metadata.details
        )
        new_loss_details = tlz.keyfilter(
            lambda key: key in attributes_list, cell.metadata.loss_details
        )
        return Metadata(
            risk_basis=(
                cell.metadata.risk_basis if "risk_basis" in attributes_list else None
            ),
            reinsurance_basis=(
                cell.metadata.reinsurance_basis
                if "reinsurance_basis" in attributes_list
                else None
            ),
            country=cell.metadata.country if "country" in attributes_list else None,
            currency=cell.metadata.currency if "currency" in attributes_list else None,
            loss_definition=(
                cell.metadata.loss_definition
                if "loss_definition" in attributes_list
                else None
            ),
            per_occurrence_limit=(
                cell.metadata.per_occurrence_limit
                if "per_occurrence_limit" in attributes_list
                else None
            ),
            details=new_details,
            loss_details=new_loss_details,
        )

    metadata_patched_cells = [
        cell.replace(metadata=metadata_mod(cell)) for cell in triangle
    ]
    return Triangle(metadata_patched_cells)
