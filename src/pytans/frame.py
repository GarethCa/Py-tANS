"""Self-contained compressed frame format.

:func:`compress` produces a byte string that carries everything needed to
decompress it: the normalised symbol table, the original length, and the
tANS payload. Inputs that would not shrink (already-compressed or random
data, tiny inputs) are stored verbatim behind the same header, so
``decompress(compress(x)) == x`` for any bytes ``x``.

Frame layout (all multi-byte ints big-endian)::

    0..3   magic  b"tANS"
    4      format version (1)
    5      mode: 0 = stored/raw, 1 = tANS-compressed

    raw mode:   uvarint(original length), raw bytes
    tans mode:  table_log (1 byte)
                pad bits in final payload byte, 0-7 (1 byte)
                number of distinct symbols - 1 (1 byte)
                per symbol: byte value (1) + normalised count (2)
                uvarint(original length)
                uvarint(payload byte length), payload
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

from .coder import BytesLike, TansCoder
from .exceptions import CorruptedDataError
from .tables import optimal_table_log
from . import tables

MAGIC = b"tANS"
VERSION = 1
_MODE_RAW = 0
_MODE_TANS = 1
_MODE_STREAM = 2  # sequence of block frames; see pytans.stream


def _write_uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _read_uvarint(blob: bytes, pos: int) -> Tuple[int, int]:
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


def _take(blob: bytes, pos: int, n: int) -> Tuple[bytes, int]:
    if pos + n > len(blob):
        raise CorruptedDataError("truncated frame")
    return blob[pos : pos + n], pos + n


def compress(
    data: BytesLike,
    table_log: Optional[int] = None,
    max_table_log: int = tables.DEFAULT_MAX_TABLE_LOG,
) -> bytes:
    """Compress ``data`` into a self-contained frame.

    ``table_log`` forces a specific state-table size; by default one is
    chosen from the input (capped at ``max_table_log``). Falls back to
    storing the input verbatim when compression would not help.
    """
    data = bytes(data)
    raw_frame = MAGIC + bytes((VERSION, _MODE_RAW)) + _write_uvarint(len(data)) + data
    if len(data) < 2:
        return raw_frame

    counts = Counter(data)
    if table_log is None:
        table_log = optimal_table_log(len(data), len(counts), max_table_log)
    coder = TansCoder(counts, table_log)
    payload, bit_length = coder.encode(data)
    pad = len(payload) * 8 - bit_length

    symbol_table = bytearray()
    for symbol, count in coder.normalized_counts.items():
        symbol_table.append(symbol)
        symbol_table += count.to_bytes(2, "big")

    tans_frame = (
        MAGIC
        + bytes((VERSION, _MODE_TANS, coder.table_log, pad, len(coder.normalized_counts) - 1))
        + bytes(symbol_table)
        + _write_uvarint(len(data))
        + _write_uvarint(len(payload))
        + payload
    )
    return tans_frame if len(tans_frame) < len(raw_frame) else raw_frame


def decompress(blob: BytesLike) -> bytes:
    """Decompress a frame produced by :func:`compress`."""
    blob = bytes(blob)
    header, pos = _take(blob, 0, 6)
    if header[:4] != MAGIC:
        raise CorruptedDataError("not a pytans frame (bad magic)")
    version, mode = header[4], header[5]
    if version != VERSION:
        raise CorruptedDataError(f"unsupported frame version {version}")

    if mode == _MODE_RAW:
        length, pos = _read_uvarint(blob, pos)
        data, pos = _take(blob, pos, length)
        if pos != len(blob):
            raise CorruptedDataError("trailing bytes after frame")
        return data

    if mode != _MODE_TANS:
        raise CorruptedDataError(f"unknown frame mode {mode}")

    fields, pos = _take(blob, pos, 3)
    table_log, pad, n_symbols = fields[0], fields[1], fields[2] + 1
    if pad > 7:
        raise CorruptedDataError("invalid padding length")

    counts = {}
    entries, pos = _take(blob, pos, 3 * n_symbols)
    for i in range(n_symbols):
        symbol = entries[3 * i]
        counts[symbol] = int.from_bytes(entries[3 * i + 1 : 3 * i + 3], "big")
    if len(counts) != n_symbols:
        raise CorruptedDataError("duplicate symbol in frame symbol table")
    if (
        not tables.MIN_TABLE_LOG <= table_log <= tables.MAX_TABLE_LOG
        or sum(counts.values()) != 1 << table_log
    ):
        raise CorruptedDataError("symbol table inconsistent with table_log")

    length, pos = _read_uvarint(blob, pos)
    payload_len, pos = _read_uvarint(blob, pos)
    payload, pos = _take(blob, pos, payload_len)
    if pos != len(blob):
        raise CorruptedDataError("trailing bytes after frame")

    coder = TansCoder(counts, table_log)
    return coder.decode(payload, payload_len * 8 - pad, length)
