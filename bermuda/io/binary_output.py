import datetime
import gzip
import math
import struct
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import awswrangler as wr
import numpy as np

from ..base import Cell, CumulativeCell, IncrementalCell, Metadata, MetadataValue
from ..triangle import Triangle
from .binary import *

__all__ = [
    "triangle_to_binary",
]


# Here's a quick crash course on binary file formats in general, and Bermuda binary in particular.
#
# All of this commentary is going to be written from the perspective of writing a binary file,
# but it applies just as well to reading (except in reverse, of course). The writer and reader
# are near-exact mirror images of each other -- for every _write_* function in this file, there's
# a corresponding inverse _read_* function in binary_input.py. Reading both files at the same
# time in adjacent windows is not a bad idea (and in fact is how I wrote them).
#
# Bermuda binary is a byte-aligned format, meaning that every token that we're going to be writing
# is a multiple of 8 bits. We could get a little bit of space savings by only using exactly the
# number of bits we need at each point, but only at the cost of quite a lot of additional
# complexity. Conceptually, we can think of the files an array of bytes (which is not the same
# thing as a string!). We just need a way to convert triangles and their constituent types to
# bytearrays and vice versa.
#
# We rely on the struct package to convert primitives (like integers) to or from bytearrays.
# Unlike vanilla Python, we don't have (easy) access to infinite-precision integers. For the
# sake of efficiency, we're using fixed-width types for all integer values, which means that
# there are limits on what can be serialized. For example, strings must be less than 32,768
# characters long (unless they contain non-ASCII characters, where the limit is lower). In
# general, the file format size limits are set high enough that I can't conceive of any future
# non-perverse use case that would exceed them.
#
# We specify little-endian byte ordering for (almost) everything that we serialize. Unfortunately,
# that's not sufficient to ensure cross-platform compatibility, as arrays are written with
# platform-native byte ordering performance reasons. However, x86, x86-64, and ARM (i.e. M1)
# are all little-endian, so we should be safe until we start trying to run Bermuda on a
# mainframe with IBM AIX processors.
#
# Higher-order data types are straightforward extensions of primitives. We serialize strings
# by writing a two-byte field with the number of bytes in the UTF-8-encoded string, followed
# by the UTF-8-encoded string itself. We serialize dictionaries by writing a series of key-value
# pairs, followed by a special byte that indicates the end of the dictionary. We serialize
# polymorphic types (such as dictionary values, which can be arrays, strings, floats, ints,
# booleans, etc) by writing a special byte that indicates the data type, followed by the binary
# representation of the value itself. We serialize dates by writing out the year, month, and day
# as fixed-width integers. We serialize objects (such as Metadata and Cell) by writing out all
# of their fields in a specified order.
#
# The Bermuda binary layout is pretty straightforward. Writing a binary consists of the following
# steps:
#   - Write the magic number (see the Wikipedia article at File_format#Magic_number
#   - Write the version number of the file format (which is 1 for now; if we ever need to
#     change the format, this allows us to preserve backwards compatibility)
#   - Write the string pool. This is the set of all distinct strings that show up as keys in
#     cell.values, metadata.details, or metadata.loss_details. This lets us save space when
#     writing dictionaries. Instead of writing out the string "earned_premium" hundreds of times,
#     we can simply write the index in the string pool of the "earned_premium" entry each time.
#   - For each cell in the triangle, see if the cell has the same metadata as the previous cell.
#     (If the cell is the first cell, assume the previous cell metadata is None.) If the metadata
#     is different, write a metadata indicator byte, then write the contents of the new metadata.
#     Regardless, then write a byte that indicates the type of cell, followed by the contents of
#     the cell except for the metadata. If cell is not immediately preceded by metadata, we can
#     assume that the cell's metadata is the same as the previous cell.


def triangle_to_binary(
    triangle: Triangle, filename: str, compress: bool = False
) -> None:
    """Convert a Bermuda triangle to a binary file format.

    Args:
        triangle: The Bermuda triangle to save out.
        filename: The filename to save the triangle to.
        compress: Whether to compress the file.
    """
    # Make sure the filename extension follows our naming conventions.
    filepath = Path(filename).expanduser()
    extension = filepath.suffix

    if compress and extension != ".tribc":
        warnings.warn(
            "Compressed Bermuda binaries should be saved with the `tribc` extension"
        )
    elif not compress and extension != ".trib":
        warnings.warn(
            "Uncompressed Bermuda binaries should be saved with the `trib` extension"
        )

    # Write the triangle to the appropriate kind of file
    if filename.startswith("s3:"):
        with NamedTemporaryFile() as temp:
            _write_binary(triangle, temp.name, compress)
            wr.s3.upload(local_file=temp.name, path=filename)
    else:
        _write_binary(triangle, filepath, compress)


def _write_binary(triangle, filepath, compress):
    if compress:
        with gzip.open(filepath, "wb", compresslevel=5) as outfile:
            _write_triangle(triangle, outfile)
    else:
        with open(filepath, "wb") as outfile:
            _write_triangle(triangle, outfile)


def _write_triangle(triangle: Triangle, stream):
    # Write the file header
    stream.write(MAGIC)
    stream.write(VERSION)

    # Write the string pool
    pool_lookup = _write_string_pool(triangle, stream)

    # Write the triangle contents
    prev_metadata = None
    for cell in triangle:
        # Only need to write the metadata if it's different from the previous cell
        if prev_metadata != cell.metadata:
            _write_metadata(cell.metadata, stream, pool_lookup)
        _write_cell(cell, stream, pool_lookup)
        prev_metadata = cell.metadata


def _write_string_pool(triangle: Triangle, stream) -> dict[str, int]:
    # Collect all of the keys used in all of the dictionaries
    all_keys = set(triangle.fields)
    for metadata in triangle.metadata:
        all_keys |= set(metadata.details.keys()) | set(metadata.loss_details.keys())
    # Explicitly sorting here so the hash of a Bermuda binary file for a fixed set of contents
    # will be invariant.
    all_sorted_keys = sorted(all_keys)

    # Write out the length of the string pool
    stream.write(struct.pack("<h", len(all_sorted_keys)))

    # Write out the constituent strings and construct a string-to-index lookup table
    pool_lookup = dict()
    for ndx, key in enumerate(all_sorted_keys):
        _write_string(key, stream)
        pool_lookup[key] = ndx

    return pool_lookup


def _write_cell(cell: Cell, stream, pool_lookup: dict[str, int]):
    # Write the byte that indicates what type of Cell it is
    if isinstance(cell, CumulativeCell):
        stream.write(CUMULATIVE_CELL)
    elif isinstance(cell, IncrementalCell):
        stream.write(INCREMENTAL_CELL)
    else:
        stream.write(CELL)

    # Write the cell contents
    _write_date(cell.period_start, stream)
    _write_date(cell.period_end, stream)
    _write_date(cell.evaluation_date, stream)
    _write_dict(cell.values, stream, pool_lookup)
    if isinstance(cell, IncrementalCell):
        _write_date(cell.prev_evaluation_date, stream)


def _write_metadata(metadata: Metadata, stream, pool_lookup: dict[str, int]):
    # Write the metadata indicator byte
    stream.write(METADATA)

    # Write all of the metadata contents
    _write_string(metadata.risk_basis, stream)
    _write_string(metadata.country, stream)
    _write_string(metadata.currency, stream)
    _write_string(metadata.reinsurance_basis, stream)
    _write_string(metadata.loss_definition, stream)
    _write_float(metadata.per_occurrence_limit, stream)
    _write_dict(metadata.details, stream, pool_lookup)
    _write_dict(metadata.loss_details, stream, pool_lookup)


def _write_string(string: str | None, stream):
    # Special handling for None values
    if string is None:
        stream.write(struct.pack("<h", -1))
    else:
        # Implicit encoding to UTF-8 here
        string_bytes = string.encode()
        # Write the number of bytes -- not guaranteed to be the same as len(string)!
        stream.write(struct.pack("<H", len(string_bytes)))
        # Write the string contents
        stream.write(string_bytes)


def _write_date(date: datetime.date, stream):
    # Signed 2-byte for year, unsigned 1-byte for month and day
    stream.write(struct.pack("<hBB", date.year, date.month, date.day))


def _write_float(num: float | None, stream):
    # Use nan to represent Nones
    if num is None:
        stream.write(struct.pack("<d", math.nan))
    else:
        stream.write(struct.pack("<d", num))


def _write_array(array: np.ndarray, stream):
    # Write a byte indicating the dtype of the array
    if array.dtype == "float64":
        stream.write(FLOAT_ARRAY)
    elif array.dtype == "int64":
        stream.write(INT_ARRAY)
    else:
        raise ValueError(f"Can't serialize a NumPy array of dtype {array.dtype}")

    # Write the number of dimensions in the array
    num_dims = len(array.shape)
    stream.write(struct.pack("<B", num_dims))
    # Write each of the dimensions in the array
    for dim in array.shape:
        stream.write(struct.pack("<L", dim))
    # Write the array values
    # Not using array.tofile() because NumPy uses seek() internally, which doesn't
    # play nicely with compression
    stream.write(array.tobytes())


def _write_dict(dct: dict[str, MetadataValue], stream, pool_lookup: dict[str, int]):
    # Write each of the key-value pairs
    for k, v in dct.items():
        # Write the index of the key in the string pool
        stream.write(struct.pack("<H", pool_lookup[k]))
        # Write the value
        _write_generic_value(v, stream)
    # Add the end-of-dict byte marker
    stream.write(DICT_END)


def _write_generic_value(val: MetadataValue, stream):
    # Figure out the data type, write a byte indicating the value type, then write the
    # value itself.
    if isinstance(val, str):
        stream.write(STRING)
        _write_string(val, stream)
    elif isinstance(val, bool):
        stream.write(BOOL)
        stream.write(struct.pack("?", val))
    elif isinstance(val, (int, np.int64)):
        stream.write(INT)
        stream.write(struct.pack("<q", val))
    elif isinstance(val, float):
        stream.write(FLOAT)
        stream.write(struct.pack("<d", val))
    elif isinstance(val, np.ndarray):
        _write_array(val, stream)
    elif isinstance(val, datetime.date):
        stream.write(DATE)
        _write_date(val, stream)
    else:
        stream.write(NONE)
