from ..triangle import Triangle, TriangleSlice

__all__ = [
    "slice_to_triangle",
    "triangle_to_slice",
]


def slice_to_triangle(triangle_slice):
    return Triangle(triangle_slice.cells)


def triangle_to_slice(triangle):
    return TriangleSlice(triangle.cells)
