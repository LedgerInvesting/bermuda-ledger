import datetime
import json
import warnings
from typing import IO, Any, Union

import numpy as np

from ..base import Cell, CumulativeCell, IncrementalCell, Metadata
from ..triangle import Triangle

__all__ = [
    "json_to_triangle",
    "json_string_to_triangle",
    "triangle_json_loads",
    "triangle_json_load",
    "triangle_to_json",
]


def json_to_triangle(
    file_or_fname: Union[str, IO], date_format: str = "%Y-%m-%d"
) -> Triangle:
    if isinstance(file_or_fname, str):
        with open(file_or_fname, "r") as infile:
            cells = json.load(infile, cls=TriangleDecoder, date_format=date_format)
    else:
        cells = json.load(file_or_fname, cls=TriangleDecoder, date_format=date_format)
    return Triangle(cells)


def json_string_to_triangle(string: str, date_format: str = "%Y-%m-%d") -> Triangle:
    cells = json.loads(string, cls=TriangleDecoder, date_format=date_format)
    return Triangle(cells)


def triangle_json_loads(string: str, date_format: str = "%Y-%m-%d") -> Triangle:
    warnings.warn(
        "triangle_json_loads is being deprecated, use json_string_to_triangle instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return json_string_to_triangle(string, date_format)


def triangle_json_load(file: IO, date_format: str = "%Y-%m-%d") -> Triangle:
    warnings.warn(
        "triangle_json_load is being deprecated, use json_to_triangle instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return json_to_triangle(file, date_format)


def triangle_to_json(
    tri: Triangle[Cell], file_or_fname: Union[str, IO] | None = None
) -> str | None:
    """Convert a triangle to JSON.

    Args:
        tri: A triangle to convert to a JSON representation.
        file_or_fname: A filename, a file handle, or `None`.
    Returns:
        If `file_or_fname` is `None`, then a string with the JSON representation of the
        triangle is returned. Otherwise, the JSON representation is written to the provided
        file and `None` is returned.
    """
    if file_or_fname:
        # The file_or_fname behavior is the same as triangle_to_wide_csv and others.
        if isinstance(file_or_fname, str):
            with open(file_or_fname, "w") as outfile:
                json.dump(tri, outfile, cls=TriangleEncoder)
        else:
            json.dump(tri, file_or_fname, cls=TriangleEncoder)
    else:
        # The return-a-string behavior is to maintain backwards compatability with older
        # versions of Bermuda that didn't support directly writing JSON to a file.
        return json.dumps(tri, cls=TriangleEncoder)


class TriangleDecoder(json.JSONDecoder):
    def __init__(self, date_format="%Y-%m-%d", *args, **kwargs):
        self.date_format = date_format
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "slices" in obj:
            return sum(obj["slices"], [])
        if "cells" in obj:
            return self._parse_cell_set(obj)
        if "period_start" in obj and "period_end" in obj and "values" in obj:
            return self._parse_observation(obj)
        return obj

    @staticmethod
    def _parse_cell_set(obj):
        metadata = Metadata(
            risk_basis=obj.get("risk_basis", "Accident"),
            country=obj.get("country"),
            currency=obj.get("currency"),
            reinsurance_basis=obj.get("reinsurance_basis"),
            loss_definition=obj.get("loss_definition"),
            per_occurrence_limit=obj.get("per_occurrence_limit"),
            details=obj.get("details", {}),
            loss_details=obj.get("loss_details", {}),
        )

        return [ob.replace(metadata=metadata) for ob in obj["cells"]]

    def _parse_observation(self, obj):
        values = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in obj["values"].items()
        }
        if "prev_evaluation_date" in obj.keys():
            cell = IncrementalCell(
                period_start=datetime.datetime.strptime(
                    obj["period_start"], self.date_format
                ).date(),
                period_end=datetime.datetime.strptime(
                    obj["period_end"], self.date_format
                ).date(),
                evaluation_date=datetime.datetime.strptime(
                    obj["evaluation_date"], self.date_format
                ).date(),
                prev_evaluation_date=datetime.datetime.strptime(
                    obj["prev_evaluation_date"], self.date_format
                ).date(),
                values=values,
            )
        else:
            cell = CumulativeCell(
                period_start=datetime.datetime.strptime(
                    obj["period_start"], self.date_format
                ).date(),
                period_end=datetime.datetime.strptime(
                    obj["period_end"], self.date_format
                ).date(),
                evaluation_date=datetime.datetime.strptime(
                    obj["evaluation_date"], self.date_format
                ).date(),
                values=values,
            )
        return cell


class TriangleEncoder(json.JSONEncoder):
    def default(self, obj):
        return {"slices": [_slice_to_dict(slice_) for slice_ in obj.slices.values()]}


def _slice_to_dict(slice_: Triangle[Cell]) -> dict[str, Any]:
    return {
        **{
            k: v
            for k, v in slice_.cells[0].metadata.as_dict().items()
            if v is not None and v != {}
        },
        "cells": [_cell_to_dict(cell) for cell in slice_.cells],
    }


def _cell_to_dict(cell: Cell) -> dict[str, Any]:
    base_dict = {
        "period_start": cell.period_start.strftime("%Y-%m-%d"),
        "period_end": cell.period_end.strftime("%Y-%m-%d"),
        "evaluation_date": cell.evaluation_date.strftime("%Y-%m-%d"),
    }
    values_dict = {
        "values": {
            field: value.tolist() if isinstance(value, np.ndarray) else value
            for field, value in cell.values.items()
        },
    }
    if isinstance(cell, IncrementalCell):
        base_dict.update(
            {"prev_evaluation_date": cell.prev_evaluation_date.strftime("%Y-%m-%d")}
        )

    return {**base_dict, **values_dict}
