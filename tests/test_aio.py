import asyncio
import io
import random

import pytest

import pytans
from pytans import CorruptedDataError, aio


def make_compressible(n, seed=0):
    rng = random.Random(seed)
    return bytes(rng.choices(b"abcde \n", weights=[50, 20, 10, 8, 4, 7, 1], k=n))


class AsyncReader:
    """aiofiles-style async source over in-memory bytes."""

    def __init__(self, data):
        self._bio = io.BytesIO(data)

    async def read(self, n=-1):
        return self._bio.read(n)


class AsyncWriter:
    """Async sink with an awaitable write, plus a drain() like StreamWriter."""

    def __init__(self):
        self.buffer = io.BytesIO()
        self.drained = 0

    async def write(self, data):
        self.buffer.write(data)

    async def drain(self):
        self.drained += 1


def test_async_roundtrip():
    data = make_compressible(50_000)

    async def run():
        packed = AsyncWriter()
        await aio.compress_stream(AsyncReader(data), packed, block_size=8192)
        restored = AsyncWriter()
        counts = await aio.decompress_stream(
            AsyncReader(packed.buffer.getvalue()), restored
        )
        return packed, restored, counts

    packed, restored, (bytes_read, bytes_written) = asyncio.run(run())
    assert restored.buffer.getvalue() == data
    assert bytes_written == len(data)
    assert bytes_read == len(packed.buffer.getvalue())
    assert packed.drained > 0  # drain() was honoured for backpressure


def test_async_with_asyncio_streamreader():
    data = make_compressible(20_000)

    async def run():
        packed = AsyncWriter()
        await aio.compress_stream(AsyncReader(data), packed, block_size=4096)

        reader = asyncio.StreamReader()
        reader.feed_data(packed.buffer.getvalue())
        reader.feed_eof()
        restored = AsyncWriter()
        await aio.decompress_stream(reader, restored)
        return restored.buffer.getvalue()

    assert asyncio.run(run()) == data


def test_async_accepts_sync_file_objects():
    # Duck typing: plain BytesIO works too, awaitable or not.
    data = make_compressible(10_000)

    async def run():
        packed = io.BytesIO()
        await aio.compress_stream(io.BytesIO(data), packed, block_size=2048)
        packed.seek(0)
        restored = io.BytesIO()
        await aio.decompress_stream(packed, restored)
        return restored.getvalue()

    assert asyncio.run(run()) == data


def test_async_accepts_single_oneshot_frame():
    data = make_compressible(5_000)

    async def run():
        restored = AsyncWriter()
        await aio.decompress_stream(AsyncReader(pytans.compress(data)), restored)
        return restored.buffer.getvalue()

    assert asyncio.run(run()) == data


def test_async_corruption_detected():
    data = make_compressible(20_000)

    async def run():
        packed = io.BytesIO()
        await aio.compress_stream(io.BytesIO(data), packed, block_size=4096)
        blob = bytearray(packed.getvalue())
        blob[len(blob) // 2] ^= 0x10
        await aio.decompress_stream(AsyncReader(bytes(blob)), AsyncWriter())

    with pytest.raises(CorruptedDataError):
        asyncio.run(run())


def test_async_truncation_detected():
    data = make_compressible(20_000)

    async def run(cut):
        packed = io.BytesIO()
        await aio.compress_stream(io.BytesIO(data), packed, block_size=4096)
        await aio.decompress_stream(
            AsyncReader(packed.getvalue()[:cut]), AsyncWriter()
        )

    for cut in (3, 10, 1000):
        with pytest.raises(CorruptedDataError):
            asyncio.run(run(cut))
