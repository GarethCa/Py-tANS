"""LEB128-style unsigned varints, shared by the frame and transform layers."""

from __future__ import annotations

from typing import Tuple

from .exceptions import CorruptedDataError


def write_uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def read_uvarint(blob: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(blob) or shift > 63:
            raise CorruptedDataError("truncated or invalid frame")
        byte = blob[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
