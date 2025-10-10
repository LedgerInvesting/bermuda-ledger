import datetime
import gzip
import io
import math
import struct
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import awswrangler as wr
import boto3
import numpy as np
from botocore.config import Config

from ..base import Cell, CumulativeCell, IncrementalCell, Metadata, MetadataValue
from ..triangle import Triangle
from .binary import *

__all__ = [
    "binary_to_triangle",
]


# See binary_output.py for more extensive comments on the design of the file format.
_S3 = boto3.client(
    "s3",
    config=Config(
        max_pool_connections=64,
        retries={"max_attempts": 10, "mode": "standard"},
        tcp_keepalive=True,
    ),
)


def binary_to_triangle(filename: str, compress: bool | None = None) -> Triangle:
    """Read a Bermuda triangle from Bermuda's binary file format.

    Args:
        filename: The filename to read the triangle from.
        compress: Whether the source file is compressed. If not provided explicitly, compression
            will be inferred from the file extension.
    """
    filepath = Path(filename).expanduser()
    extension = filepath.suffix
    # Try to deduce compression status if not provided explicitly
    if not compress:
        if extension == ".trib":
            compress = False
        elif extension == ".tribc":
            compress = True
        else:
            raise ValueError(
                f"Can't infer compression status on a triangle saved with {extension} extension"
            )
    # Check if compression status and extension are compatible if provided explicitly
    else:
        if compress and extension != ".tribc":
            warnings.warn(
                "Compressed Bermuda binaries should be saved with the `tribc` extension"
            )
        elif not compress and extension != ".trib":
            warnings.warn(
                "Uncompressed Bermuda binaries should be saved with the `trib` extension"
            )

    # Read the triangle from the appropriate kind of file
    if filename.startswith("s3:"):
        # Stream directly from S3
        raw = _open_s3_stream(filename)
        try:
            if compress:
                # Decompress as a stream; buffer again so .peek() is reliable downstream
                with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                    stream = io.BufferedReader(gz, buffer_size=64 * 1024)
                    return _read_triangle(stream)
            else:
                return _read_triangle(raw)
        finally:
            # Make sure to close the StreamingBody/BufferedReader
            raw.close()
    else:
        return _read_binary(filepath, compress)


class _BodyRawIO(io.RawIOBase):
    """Adapter so BufferedReader can wrap S3 StreamingBody."""

    def __init__(self, body):
        self.body = body

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        # Read up to len(b) bytes from the StreamingBody
        data = self.body.read(len(b))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        return n

    def read(self, n: int = -1) -> bytes:
        # Optional, BufferedReader uses readinto() primarily
        return self.body.read(None if n is None or n < 0 else n)

    def close(self):
        try:
            self.body.close()
        finally:
            super().close()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    # supports s3://bucket/key and s3:bucket/key
    u = uri.replace("s3://", "s3:", 1)
    _, rest = u.split("s3:", 1)
    bucket, key = rest.lstrip("/").split("/", 1)
    return bucket, key


def _open_s3_stream(s3_uri: str) -> io.BufferedReader:
    bucket, key = _parse_s3_uri(s3_uri)
    resp = _S3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]  # botocore.response.StreamingBody
    # Wrap in BufferedReader so .peek() works and small reads are efficient
    return io.BufferedReader(_BodyRawIO(body), buffer_size=64 * 1024)


def _read_binary(filepath, compress):
    if compress:
        with gzip.open(filepath, "rb", compresslevel=5) as infile:
            return _read_triangle(infile)
    else:
        with open(filepath, "rb") as infile:
            return _read_triangle(infile)


def _read_triangle(stream) -> Triangle:
    # Make sure the file has the right magic and version number
    if stream.read(4) != MAGIC:
        raise ValueError("This file is not a Bermuda triangle blob")
    if stream.read(1) != VERSION:
        raise ValueError(
            "The version of this file is not supported by this version of Bermuda"
        )

    # Read the string pool
    string_pool = _read_string_pool(stream)

    # Read the triangle contents
    current_metadata = None
    cells = []
    while True:
        marker = stream.read(1)
        # Only need to read the metadata if it's explicitly declared
        if marker == METADATA:
            current_metadata = _read_metadata(stream, string_pool)
        elif marker == CELL or marker == CUMULATIVE_CELL or marker == INCREMENTAL_CELL:
            cells.append(_read_cell(stream, marker, current_metadata, string_pool))
        # stream.read(1) will return an empty array if we're at EOF
        else:
            break

    return Triangle(cells)


def _read_string_pool(stream) -> list[str]:
    # Read the length of the string pool
    num_elements = struct.unpack("<H", stream.read(2))[0]
    # Construct an array with all of the string pool elements
    return [_read_string(stream) for _ in range(num_elements)]


def _read_cell(
    stream, marker: bytes, metadata: Metadata, string_pool: list[str]
) -> Cell:
    # Construct the appropriate type of cell based on the Cell-type marker byte
    if marker == CUMULATIVE_CELL:
        return CumulativeCell(
            period_start=_read_date(stream),
            period_end=_read_date(stream),
            evaluation_date=_read_date(stream),
            values=_read_dict(stream, string_pool),
            metadata=metadata,
        )
    elif marker == INCREMENTAL_CELL:
        return IncrementalCell(
            period_start=_read_date(stream),
            period_end=_read_date(stream),
            evaluation_date=_read_date(stream),
            values=_read_dict(stream, string_pool),
            prev_evaluation_date=_read_date(stream),
            metadata=metadata,
        )
    else:
        return Cell(
            period_start=_read_date(stream),
            period_end=_read_date(stream),
            evaluation_date=_read_date(stream),
            values=_read_dict(stream, string_pool),
            metadata=metadata,
        )


def _read_metadata(stream, string_pool: list[str]) -> Metadata:
    # Construct a Metadata object from the serialized fields
    return Metadata(
        risk_basis=_read_string(stream),
        country=_read_string(stream),
        currency=_read_string(stream),
        reinsurance_basis=_read_string(stream),
        loss_definition=_read_string(stream),
        per_occurrence_limit=_read_float(stream),
        details=_read_dict(stream, string_pool),
        loss_details=_read_dict(stream, string_pool),
    )


def _read_string(stream) -> str | None:
    # Read the string length
    length = struct.unpack("<h", stream.read(2))[0]
    # Special handling for None values
    if length == -1:
        return None
    return stream.read(length).decode("utf-8")


def _read_date(stream) -> datetime.date:
    year, month, day = struct.unpack("<hBB", stream.read(4))
    return datetime.date(year, month, day)


def _read_float(stream) -> float | None:
    num = struct.unpack("<d", stream.read(8))[0]
    # Special handling for None values
    if math.isnan(num):
        return None
    return num


def _read_array(stream, dtype: np.dtype) -> np.ndarray:
    # Read the number of dimensions
    num_dims = struct.unpack("<B", stream.read(1))[0]
    # Read the array shape
    shape = tuple([struct.unpack("<L", stream.read(4))[0] for _ in range(num_dims)])

    # Figure out the total number of items in the array
    num_items = 1
    for dim in shape:
        num_items *= dim

    # Read the array values
    # Not using np.fromfile() because NumPy uses seek() internally, which doesn't play
    # nicely with compression
    array_bytes = stream.read(num_items * 8)
    return np.frombuffer(array_bytes, dtype).reshape(shape)


def _read_dict(stream, string_pool: list[str]) -> dict[str, MetadataValue]:
    result = {}
    # Keep reading key-value pairs
    while stream.peek(1)[:1] != DICT_END:
        # Grab the index of the key in the string pool
        key_ndx = struct.unpack("<H", stream.read(2))[0]
        key = string_pool[key_ndx]
        # Read the value
        result[key] = _read_generic_value(stream)
    # Consume the end-of-dict byte that we must have just peeked at
    stream.read(1)
    return result


def _read_generic_value(stream) -> MetadataValue:
    # Read the value type byte
    valtype = stream.read(1)
    # Read the contextually appropriate value type, based on the marker byte
    if valtype == STRING:
        return _read_string(stream)
    elif valtype == BOOL:
        return struct.unpack("?", stream.read(1))[0]
    elif valtype == INT:
        return struct.unpack("<q", stream.read(8))[0]
    elif valtype == FLOAT:
        return struct.unpack("<d", stream.read(8))[0]
    elif valtype == INT_ARRAY:
        return _read_array(stream, np.dtype("int64"))
    elif valtype == FLOAT_ARRAY:
        return _read_array(stream, np.dtype("float64"))
    elif valtype == DATE:
        return _read_date(stream)
    elif valtype == NONE:
        return None
