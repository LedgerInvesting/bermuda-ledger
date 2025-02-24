import numpy as np
import toolz as tlz

from ..errors import TriangleError
from ..triangle import Triangle

__all__ = [
    "add_statics",
    "array_from_field",
    "array_sizes",
    "array_size",
]


def add_statics(
    triangle: Triangle,
    source: Triangle,
    statics: list[str] = ["earned_premium", "earned_exposure"],
) -> Triangle:
    """Add "static" fields to a `Triangle`.
    Some fields (for example, accident-period earned premium) can be considered as fixed
    with respect to development lag. This method takes static fields from one triangle and
    adds them to another triangle. For example, if we have a triangle of predicted `paid_loss`
    and another disjoint triangle with observed `paid_loss`, we can fill in `earned_premium`
    values in the former triangle with values from the latter triangle.
    Args:
        source: A `Triangle` that will be used to fill in static fields.
        statics: A list of field names to fill in.
    Returns:
        A `Triangle` with static fields filled in from `source`.
    """

    rich_cells = []
    source_slices = source.slices
    for key, slc in triangle.slices.items():
        source_slice = source_slices.get(key, None)
        if source_slice is not None:
            rich_cells += _add_statics_slice(slc, source_slice, statics)
        else:
            rich_cells += slc.cells
    return Triangle(rich_cells)


def _add_statics_slice(slice, source_slice, statics):
    source_indexed = tlz.valmap(
        lambda row: sorted(row, key=lambda ob: ob.evaluation_date)[-1],
        tlz.groupby(lambda cell: cell.period, source_slice.cells),
    )

    rich_cells = []
    for cell in slice:
        source_cell = source_indexed.get(cell.period, None)
        if source_cell is not None:
            rich_cells.append(cell.add_statics(source_cell, statics))
        else:
            rich_cells += [cell]

    return rich_cells


def array_from_field(triangle: Triangle, field: str) -> np.ndarray:
    """Return an array with cell-level values from the triangle.

    Args:
        tri: Triangle to pull the cell values from.
        field: Name of the field to turn into an array.
    Returns:
        An array of dimension (N, K) where N is the number of cells in the source triangle
            and K is the array_size of the field.
    """

    if not all([field in cell for cell in triangle]):
        raise TriangleError(
            f"Cannot construct an array of {field} because the field is not present in every cell"
        )

    target_size = array_size(triangle)
    consistent_values = []
    for cell in triangle.cells:
        value = np.array(cell[field])
        if value.size != target_size:
            value = np.full((target_size,), value)
        consistent_values.append(value)
    return np.array(consistent_values)


def array_sizes(triangle: Triangle) -> list[int]:
    """Get the ordered set of all distinct non-scalar array sizes in field
    values in the Triangle."""

    sizes = set()
    for cell in triangle:
        for field in cell.values.values():
            if isinstance(field, np.ndarray):
                sizes |= {field.size}

    return sorted(sizes - {1})


def array_size(triangle: Triangle) -> int:
    """Get the consistent non-scalar array size for field values in the Triangle.
    If there is more than one distinct array size, raise an error."""

    sizes = array_sizes(triangle)
    if not sizes:
        return 1
    elif len(sizes) == 1:
        return sizes[0]
    else:
        raise ValueError(f"Multiple array sizes found in triangle: {sizes}")
