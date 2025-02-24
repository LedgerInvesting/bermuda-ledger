__all__ = [
    "TriangleError",
    "TriangleEmptyError",
    "TriangleWarning",
    "DuplicateCellWarning",
]


class TriangleError(Exception):
    """Error specific to misuses of `Triangle`s."""


class TriangleEmptyError(TriangleError):
    """Error when a `Triangle` instance contains no cells or slices."""


class TriangleWarning(Warning):
    """Base class for all warnings from `Triangle`."""


class DuplicateCellWarning(TriangleWarning):
    """Warning when a Triangle contains multiple cells with the same coordinates."""
