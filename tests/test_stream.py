import io
import random

import pytest

import pytans
from pytans import CorruptedDataError, compress_stream, decompress_stream


def make_compressible(n, seed=0):
    rng = random.Random(seed)
    return bytes(rng.choices(b"abcde \n", weights=[50, 20, 10, 8, 4, 7, 1], k=n))


def stream_roundtrip(data, **kwargs):
    compressed = io.BytesIO()
    bytes_read, bytes_written = compress_stream(io.BytesIO(data), compressed, **kwargs)
    assert bytes_read == len(data)
    assert bytes_written == compressed.tell()

    compressed.seek(0)
    restored = io.BytesIO()
    read_back, written_back = decompress_stream(compressed, restored)
    assert read_back == compressed.tell()
    assert written_back == len(data)
    assert restored.getvalue() == data
    return compressed.getvalue()


def test_empty_stream():
    blob = stream_roundtrip(b"")
    assert len(blob) == 7  # header + terminator


@pytest.mark.parametrize("block_size", [1, 100, 4096, 1 << 20])
def test_roundtrip_block_sizes(block_size):
    stream_roundtrip(make_compressible(10_000), block_size=block_size)


def test_roundtrip_exact_block_multiple():
    stream_roundtrip(make_compressible(8 * 1024), block_size=1024)


def test_multiblock_compresses(tmp_path):
    data = make_compressible(500_000)
    blob = stream_roundtrip(data, block_size=64 * 1024)
    assert len(blob) < len(data) // 2


def test_short_reads_are_accumulated():
    class DribbleReader(io.BytesIO):
        """Returns at most 7 bytes per read, like a slow socket."""

        def read(self, n=-1):
            return super().read(min(n, 7) if n and n > 0 else 7)

    data = make_compressible(5_000)
    out = io.BytesIO()
    compress_stream(DribbleReader(data), out, block_size=1024)
    out.seek(0)
    restored = io.BytesIO()
    decompress_stream(out, restored)
    assert restored.getvalue() == data


def test_table_log_is_passed_through():
    stream_roundtrip(make_compressible(10_000), table_log=8)


def test_accepts_single_oneshot_frame():
    # Anything pytans.compress() made should decode through the stream API.
    for data in (b"", b"x", make_compressible(5_000)):
        restored = io.BytesIO()
        decompress_stream(io.BytesIO(pytans.compress(data)), restored)
        assert restored.getvalue() == data


def test_bad_block_size():
    with pytest.raises(ValueError):
        compress_stream(io.BytesIO(b"x"), io.BytesIO(), block_size=0)


class TestCorruptStreams:
    def make_blob(self):
        return stream_roundtrip(make_compressible(20_000), block_size=4096)

    def test_truncation_detected_everywhere(self):
        blob = self.make_blob()
        # Mid-header, mid-length, mid-frame, and missing terminator.
        for cut in (0, 3, 5, 8, len(blob) // 2, len(blob) - 1):
            with pytest.raises(CorruptedDataError):
                decompress_stream(io.BytesIO(blob[:cut]), io.BytesIO())

    def test_trailing_garbage_detected(self):
        blob = self.make_blob()
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(blob + b"!"), io.BytesIO())

    def test_corrupt_block_detected(self):
        blob = bytearray(self.make_blob())
        blob[len(blob) // 2] ^= 0x10
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(bytes(blob)), io.BytesIO())

    def test_bad_magic_and_mode(self):
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(b"NOPE\x01\x02\x00"), io.BytesIO())
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(b"tANS\x01\x09\x00"), io.BytesIO())
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(b"tANS\x63\x02\x00"), io.BytesIO())

    def test_absurd_block_length_rejected(self):
        # 2 GiB block length encoded as uvarint after a valid header.
        bogus = b"tANS\x01\x02" + b"\x80\x80\x80\x80\x08"
        with pytest.raises(CorruptedDataError):
            decompress_stream(io.BytesIO(bogus), io.BytesIO())


def test_cli_streams_big_file(tmp_path):
    from pytans.cli import main

    big = tmp_path / "big.log"
    big.write_bytes(make_compressible(2_000_000))

    assert main(["compress", str(big), "--block-size", "65536"]) == 0
    packed = tmp_path / "big.log.tans"
    assert packed.stat().st_size < big.stat().st_size // 2

    original = big.read_bytes()
    big.unlink()
    assert main(["decompress", str(packed)]) == 0
    assert big.read_bytes() == original


def test_cli_removes_partial_output_on_corrupt_input(tmp_path):
    from pytans.cli import main

    bad = tmp_path / "bad.tans"
    blob = bytearray()
    src = io.BytesIO(make_compressible(20_000))
    out = io.BytesIO()
    compress_stream(src, out, block_size=4096)
    blob = bytearray(out.getvalue())
    blob[len(blob) // 2] ^= 0x10
    bad.write_bytes(bytes(blob))

    target = tmp_path / "restored"
    with pytest.raises(SystemExit):
        main(["decompress", str(bad), "-o", str(target)])
    assert not target.exists()
