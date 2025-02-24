import struct

__all__ = [
    "MAGIC",
    "VERSION",
    "STRING",
    "INT",
    "FLOAT",
    "BOOL",
    "NONE",
    "DATE",
    "INT_ARRAY",
    "FLOAT_ARRAY",
    "DICT_END",
    "METADATA",
    "CELL",
    "CUMULATIVE_CELL",
    "INCREMENTAL_CELL",
]

# Useful constants
MAGIC = struct.pack("<L", 0x0136AF)
VERSION = struct.pack("B", 0x01)

# Flags for basic data types
STRING = struct.pack("B", 0x80)
INT = struct.pack("B", 0x81)
FLOAT = struct.pack("B", 0x82)
BOOL = struct.pack("B", 0x83)
NONE = struct.pack("B", 0x84)
DATE = struct.pack("B", 0x85)
INT_ARRAY = struct.pack("B", 0x86)
FLOAT_ARRAY = struct.pack("B", 0x87)
DICT_END = struct.pack("B", 0x88)

# Flags for Higher-order data types
METADATA = struct.pack("B", 0x10)
CELL = struct.pack("B", 0x11)
CUMULATIVE_CELL = struct.pack("B", 0x12)
INCREMENTAL_CELL = struct.pack("B", 0x13)
