"""Block-based streaming compression for large files and pipes.

tANS decodes a frame last-in-first-out, so a single frame can never be
decoded incrementally. The streaming layer instead splits the input into
independent blocks and emits one self-contained frame per block::

    0..3   magic b"tANS"
    4      format version (1)
    5      mode: 2 = stream of blocks
    per block: uvarint(frame byte length), frame (see pytans.frame)
    terminator: a single zero length byte (0x00)

Memory use is bounded by the block size on both ends, each block carries
a table tuned to its own statistics, and each block is validated
independently on decode. :func:`decompress_stream` also accepts a plain
single-frame input, so it can decode anything :func:`pytans.compress`
produced.
"""

from __future__ import annotations

from typing import BinaryIO, Optional, Tuple

from .exceptions import CorruptedDataError
from .frame import (
    MAGIC,
    VERSION,
    _MODE_RAW,
    _MODE_STREAM,
    _MODE_TANS,
    _write_uvarint,
    compress,
    decompress,
)
from .tables import DEFAULT_MAX_TABLE_LOG

#: Default uncompressed block size (128 KiB). Large enough that the
#: per-block table header (<1 KiB worst case) is noise, small enough to
#: keep memory use and decode latency low.
DEFAULT_BLOCK_SIZE = 128 * 1024

#: Upper bound accepted for a single block frame when decoding, guarding
#: against absurd lengths in corrupted streams.
_MAX_BLOCK_FRAME = 1 << 30


def _read_up_to(src: BinaryIO, n: int) -> bytes:
    """Read up to ``n`` bytes, looping over short reads (pipes, sockets)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = src.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _read_exact(src: BinaryIO, n: int) -> bytes:
    data = _read_up_to(src, n)
    if len(data) != n:
        raise CorruptedDataError("truncated stream")
    return data


def _read_uvarint(src: BinaryIO) -> int:
    value = 0
    shift = 0
    while True:
        byte = _read_exact(src, 1)[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise CorruptedDataError("invalid length in stream")


def compress_stream(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    table_log: Optional[int] = None,
    max_table_log: int = DEFAULT_MAX_TABLE_LOG,
) -> Tuple[int, int]:
    """Compress ``src`` into ``dst`` block by block.

    ``src``/``dst`` are binary file-like objects (open files, pipes,
    ``io.BytesIO`` ...). Returns ``(bytes_read, bytes_written)``.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    dst.write(MAGIC + bytes((VERSION, _MODE_STREAM)))
    bytes_read, bytes_written = 0, 6
    while True:
        block = _read_up_to(src, block_size)
        if not block:
            break
        frame = compress(block, table_log, max_table_log)
        dst.write(_write_uvarint(len(frame)) + frame)
        bytes_read += len(block)
        bytes_written += len(_write_uvarint(len(frame))) + len(frame)
    dst.write(b"\x00")
    return bytes_read, bytes_written + 1


def decompress_stream(src: BinaryIO, dst: BinaryIO) -> Tuple[int, int]:
    """Decompress a stream (or a single frame) from ``src`` into ``dst``.

    Returns ``(bytes_read, bytes_written)``. Raises
    :class:`CorruptedDataError` on truncated, malformed or tampered input.
    """
    header = _read_exact(src, 6)
    if header[:4] != MAGIC:
        raise CorruptedDataError("not a pytans stream (bad magic)")
    if header[4] != VERSION:
        raise CorruptedDataError(f"unsupported version {header[4]}")
    mode = header[5]

    if mode in (_MODE_RAW, _MODE_TANS):
        # A one-shot pytans.compress() frame: by definition single-block.
        body = bytearray(header)
        while True:
            chunk = src.read(DEFAULT_BLOCK_SIZE)
            if not chunk:
                break
            body += chunk
        data = decompress(bytes(body))
        dst.write(data)
        return len(body), len(data)

    if mode != _MODE_STREAM:
        raise CorruptedDataError(f"unknown stream mode {mode}")

    bytes_read, bytes_written = 6, 0
    while True:
        frame_len = _read_uvarint(src)
        bytes_read += len(_write_uvarint(frame_len))
        if frame_len == 0:
            break
        if frame_len > _MAX_BLOCK_FRAME:
            raise CorruptedDataError("block frame length out of range")
        block = decompress(_read_exact(src, frame_len))
        dst.write(block)
        bytes_read += frame_len
        bytes_written += len(block)
    if src.read(1):
        raise CorruptedDataError("trailing data after stream terminator")
    return bytes_read, bytes_written
