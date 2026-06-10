"""Asyncio variants of the streaming API.

``pytans.aio.compress_stream`` / ``decompress_stream`` mirror
:mod:`pytans.stream` but work with async sources and sinks — anything
whose ``read``/``write``/``drain`` may (or may not) return awaitables:
``asyncio.StreamReader``/``StreamWriter``, ``aiofiles`` handles, and
plain synchronous file objects all work.

The compression itself is CPU-bound and runs inline; the win here is
interleaving with async I/O (sockets, pipes) without buffering whole
files. Wrap calls in ``loop.run_in_executor`` if you need the event loop
free during heavy blocks.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple

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
from .stream import _MAX_BLOCK_FRAME, DEFAULT_BLOCK_SIZE
from .tables import DEFAULT_MAX_TABLE_LOG

__all__ = ["compress_stream", "decompress_stream"]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _write(dst: Any, data: bytes) -> None:
    await _maybe_await(dst.write(data))
    drain = getattr(dst, "drain", None)
    if drain is not None:
        await _maybe_await(drain())


async def _read_up_to(src: Any, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = await _maybe_await(src.read(n - len(buf)))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


async def _read_exact(src: Any, n: int) -> bytes:
    data = await _read_up_to(src, n)
    if len(data) != n:
        raise CorruptedDataError("truncated stream")
    return data


async def _read_uvarint(src: Any) -> int:
    value = 0
    shift = 0
    while True:
        byte = (await _read_exact(src, 1))[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise CorruptedDataError("invalid length in stream")


async def compress_stream(
    src: Any,
    dst: Any,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    table_log: Optional[int] = None,
    max_table_log: int = DEFAULT_MAX_TABLE_LOG,
) -> Tuple[int, int]:
    """Async equivalent of :func:`pytans.compress_stream`."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    await _write(dst, MAGIC + bytes((VERSION, _MODE_STREAM)))
    bytes_read, bytes_written = 0, 6
    while True:
        block = await _read_up_to(src, block_size)
        if not block:
            break
        frame = compress(block, table_log, max_table_log)
        header = _write_uvarint(len(frame))
        await _write(dst, header + frame)
        bytes_read += len(block)
        bytes_written += len(header) + len(frame)
    await _write(dst, b"\x00")
    return bytes_read, bytes_written + 1


async def decompress_stream(src: Any, dst: Any) -> Tuple[int, int]:
    """Async equivalent of :func:`pytans.decompress_stream`."""
    header = await _read_exact(src, 6)
    if header[:4] != MAGIC:
        raise CorruptedDataError("not a pytans stream (bad magic)")
    if header[4] != VERSION:
        raise CorruptedDataError(f"unsupported version {header[4]}")
    mode = header[5]

    if mode in (_MODE_RAW, _MODE_TANS):
        body = bytearray(header)
        while True:
            chunk = await _maybe_await(src.read(DEFAULT_BLOCK_SIZE))
            if not chunk:
                break
            body += chunk
        data = decompress(bytes(body))
        await _write(dst, data)
        return len(body), len(data)

    if mode != _MODE_STREAM:
        raise CorruptedDataError(f"unknown stream mode {mode}")

    bytes_read, bytes_written = 6, 0
    while True:
        frame_len = await _read_uvarint(src)
        bytes_read += len(_write_uvarint(frame_len))
        if frame_len == 0:
            break
        if frame_len > _MAX_BLOCK_FRAME:
            raise CorruptedDataError("block frame length out of range")
        block = decompress(await _read_exact(src, frame_len))
        await _write(dst, block)
        bytes_read += frame_len
        bytes_written += len(block)
    if await _maybe_await(src.read(1)):
        raise CorruptedDataError("trailing data after stream terminator")
    return bytes_read, bytes_written
