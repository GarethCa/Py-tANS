"""Self-contained compressed frame format.

:func:`compress` produces a byte string that carries everything needed to
decompress it: the normalised symbol table, the original length, and the
tANS payload. Inputs that would not shrink (already-compressed or random
data, tiny inputs) are stored verbatim behind the same header, so
``decompress(compress(x)) == x`` for any bytes ``x``.

Frame layout (all multi-byte ints big-endian)::

    0..3   magic  b"tANS"
    4      format version (1)
    5      mode: 0 = stored/raw, 1 = tANS, 2 = stream, 3 = transformed

    raw mode:   uvarint(original length), raw bytes
    tans mode:  table_log (1 byte)
                pad bits in final payload byte, 0-7 (1 byte)
                number of distinct symbols - 1 (1 byte)
                per symbol: byte value (1) + normalised count (2)
                uvarint(original length)
                uvarint(payload byte length), payload
    transformed mode (see pytans.transforms):
                transform id (1 byte), substream count (1 byte)
                uvarint(original length)
                per substream: uvarint(byte length) + an embedded
                raw/tans frame holding the transformed substream

Mode 2 (a stream of block frames) is documented in :mod:`pytans.stream`.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ._varint import read_uvarint as _read_uvarint
from ._varint import write_uvarint as _write_uvarint
from .coder import BytesLike, TansCoder
from .exceptions import CorruptedDataError
from .tables import optimal_table_log
from . import tables, transforms

MAGIC = b"tANS"
VERSION = 1
_MODE_RAW = 0
_MODE_TANS = 1
_MODE_STREAM = 2  # sequence of block frames; see pytans.stream
_MODE_TRANSFORM = 3

#: Below this size a transform cannot pay for its framing.
_MIN_TRANSFORM_SIZE = 16


def _take(blob: bytes, pos: int, n: int) -> "tuple[bytes, int]":
    if pos + n > len(blob):
        raise CorruptedDataError("truncated frame")
    return blob[pos : pos + n], pos + n


def _compress_plain(
    data: bytes,
    table_log: Optional[int],
    max_table_log: int,
) -> bytes:
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


def _compress_transformed(
    data: bytes,
    transform: str,
    table_log: Optional[int],
    max_table_log: int,
) -> bytes:
    wire_id, encode, _ = transforms.get_transform(transform)
    body = bytearray()
    streams = encode(data)
    body += bytes((wire_id, len(streams)))
    body += _write_uvarint(len(data))
    for stream in streams:
        sub_frame = _compress_plain(stream, table_log, max_table_log)
        body += _write_uvarint(len(sub_frame)) + sub_frame
    return MAGIC + bytes((VERSION, _MODE_TRANSFORM)) + bytes(body)


def compress(
    data: BytesLike,
    table_log: Optional[int] = None,
    max_table_log: int = tables.DEFAULT_MAX_TABLE_LOG,
    transform: Optional[str] = None,
) -> bytes:
    """Compress ``data`` into a self-contained frame.

    ``table_log`` forces a specific state-table size; by default one is
    chosen per (sub)stream. ``transform`` enables an optional modeling
    stage for repetitive data (``"bwt"`` or ``"lz77"``, see
    :mod:`pytans.transforms`); the transformed frame is only kept when it
    is actually smaller than the plain one. Falls back to storing the
    input verbatim when nothing helps.
    """
    data = bytes(data)
    plain = _compress_plain(data, table_log, max_table_log)
    if transform is None:
        return plain
    transforms.get_transform(transform)  # validate the name even if unused
    if len(data) < _MIN_TRANSFORM_SIZE:
        return plain
    transformed = _compress_transformed(data, transform, table_log, max_table_log)
    return transformed if len(transformed) < len(plain) else plain


def _decompress_body(blob: bytes, mode: int, pos: int) -> "tuple[bytes, int]":
    """Decode one frame body, returning ``(data, next_pos)``."""
    if mode == _MODE_RAW:
        length, pos = _read_uvarint(blob, pos)
        return _take(blob, pos, length)

    if mode == _MODE_TANS:
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
        coder = TansCoder(counts, table_log)
        return coder.decode(payload, payload_len * 8 - pad, length), pos

    if mode == _MODE_TRANSFORM:
        header, pos = _take(blob, pos, 2)
        _, _, decode = transforms.get_transform_by_id(header[0])
        n_streams = header[1]
        length, pos = _read_uvarint(blob, pos)
        streams = []
        for _ in range(n_streams):
            sub_len, pos = _read_uvarint(blob, pos)
            sub_frame, pos = _take(blob, pos, sub_len)
            if len(sub_frame) >= 6 and sub_frame[5] not in (_MODE_RAW, _MODE_TANS):
                # No nesting: transformed substreams are always plain frames.
                raise CorruptedDataError("invalid substream mode in transformed frame")
            streams.append(decompress(sub_frame))
        data = decode(streams)
        if len(data) != length:
            raise CorruptedDataError("transformed frame decoded to wrong length")
        return data, pos

    raise CorruptedDataError(f"unknown frame mode {mode}")


def decompress(blob: BytesLike) -> bytes:
    """Decompress a frame produced by :func:`compress`."""
    blob = bytes(blob)
    header, pos = _take(blob, 0, 6)
    if header[:4] != MAGIC:
        raise CorruptedDataError("not a pytans frame (bad magic)")
    if header[4] != VERSION:
        raise CorruptedDataError(f"unsupported frame version {header[4]}")
    data, pos = _decompress_body(blob, header[5], pos)
    if pos != len(blob):
        raise CorruptedDataError("trailing bytes after frame")
    return data
