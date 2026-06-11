import io
import random

import pytest

import pytans
from pytans import CorruptedDataError, compress, compress_stream, decompress, decompress_stream
from pytans import transforms
from pytans.transforms import (
    _bwt,
    _ibwt,
    _imtf,
    _mtf,
    _zero_rle_join,
    _zero_rle_split,
    bwt_decode,
    bwt_encode,
    lz77_decode,
    lz77_encode,
)


def make_text(n, seed=0):
    rng = random.Random(seed)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "compression", "entropy", "state", "table"]
    out = []
    size = 0
    while size < n:
        w = rng.choice(words)
        out.append(w)
        size += len(w) + 1
    return " ".join(out).encode()[:n]


SAMPLES = [
    b"",
    b"a",
    b"ab",
    b"banana banana banana",
    b"a" * 5_000,
    b"ab" * 3_000,
    make_text(20_000),
    bytes(random.Random(1).randbytes(5_000)),
    bytes(range(256)) * 20,
]
IDS = ["empty", "one", "two", "banana", "runs", "pairs", "text", "noise", "cycle"]


class TestPrimitives:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_mtf_roundtrip(self, data):
        assert _imtf(_mtf(data)) == data

    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_bwt_roundtrip(self, data):
        last, primary = _bwt(data)
        assert sorted(last) == sorted(data)  # a permutation
        assert _ibwt(last, primary) == data

    def test_bwt_pure_python_path_matches(self, monkeypatch):
        data = make_text(3_000)
        expected = _bwt(data)
        monkeypatch.setattr(transforms, "_np", None)
        assert _bwt(data) == expected

    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_zero_rle_roundtrip(self, data):
        syms, runs = _zero_rle_split(data)
        assert len(runs) >= syms.count(0)  # at least one length per collapsed run
        assert _zero_rle_join(syms, runs) == data

    def test_zero_rle_long_runs(self):
        data = b"\x00" * 1000 + b"x" + b"\x00" * 255
        syms, runs = _zero_rle_split(data)
        assert _zero_rle_join(syms, runs) == data

    def test_ibwt_rejects_bad_primary(self):
        with pytest.raises(CorruptedDataError):
            _ibwt(b"abc", 5)


class TestTransformCodecs:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    @pytest.mark.parametrize("name", ["bwt", "lz77"])
    def test_encode_decode_identity(self, name, data):
        _, encode, decode = transforms.get_transform(name)
        assert decode(encode(data)) == data

    def test_lz77_overlapping_matches(self):
        for data in (b"ab" * 4_000, b"abc" * 3_000, b"a" * 10_000):
            assert lz77_decode(lz77_encode(data)) == data

    def test_lz77_long_matches_and_offsets(self):
        chunk = bytes(random.Random(2).randbytes(40_000))
        data = chunk + b"filler" * 100 + chunk  # 40 KB match at long offset
        assert lz77_decode(lz77_encode(data)) == data

    def test_bwt_wrong_stream_count(self):
        with pytest.raises(CorruptedDataError):
            bwt_decode([b"abc"])

    def test_lz77_wrong_stream_count(self):
        with pytest.raises(CorruptedDataError):
            lz77_decode([b""] * 4)

    def test_unknown_transform_name(self):
        with pytest.raises(ValueError, match="unknown transform"):
            transforms.get_transform("brotli")

    def test_unknown_transform_id(self):
        with pytest.raises(CorruptedDataError):
            transforms.get_transform_by_id(99)


class TestFrameIntegration:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    @pytest.mark.parametrize("name", ["bwt", "lz77"])
    def test_roundtrip(self, name, data):
        assert decompress(compress(data, transform=name)) == data

    @pytest.mark.parametrize("name", ["bwt", "lz77"])
    def test_transform_shrinks_repetitive_data(self, name):
        data = make_text(60_000)
        plain = compress(data)
        transformed = compress(data, transform=name)
        assert len(transformed) < len(plain) * 0.8
        assert decompress(transformed) == data

    def test_transform_never_larger_than_plain(self):
        # Transforms hurt data without repetition; the plain frame wins.
        rng = random.Random(3)
        data = bytes(rng.choices(range(8), weights=[40, 20, 10, 8, 6, 5, 4, 3], k=20_000))
        for name in ("bwt", "lz77"):
            assert len(compress(data, transform=name)) <= len(compress(data))

    def test_invalid_transform_name_raises_even_for_tiny_input(self):
        with pytest.raises(ValueError):
            compress(b"x", transform="nope")

    def test_corrupt_transformed_frames(self):
        blob = bytearray(compress(make_text(20_000), transform="bwt"))
        assert blob[5] == 3  # transformed mode actually in use
        rng = random.Random(0)
        data = make_text(20_000)
        for _ in range(20):
            corrupted = bytearray(blob)
            corrupted[rng.randrange(8, len(blob) - 1)] ^= 1 << rng.randrange(8)
            try:
                result = decompress(bytes(corrupted))
            except CorruptedDataError:
                continue
            assert result != data  # never silently wrong

    def test_no_nested_transform_frames(self):
        # Hand-craft a mode-3 frame whose substream is itself mode 3.
        inner = compress(make_text(1_000), transform="bwt")
        assert inner[5] == 3
        from pytans._varint import write_uvarint
        body = bytes((1, 2)) + write_uvarint(1_000) + write_uvarint(len(inner)) + inner
        evil = b"tANS\x01\x03" + body
        with pytest.raises(CorruptedDataError):
            decompress(evil)


class TestStreamAndCli:
    def test_stream_roundtrip_with_transform(self):
        data = make_text(300_000)
        out = io.BytesIO()
        compress_stream(io.BytesIO(data), out, block_size=64 * 1024, transform="bwt")
        out.seek(0)
        restored = io.BytesIO()
        decompress_stream(out, restored)
        assert restored.getvalue() == data
        assert out.tell() < len(compress(data)) * 0.8

    def test_async_passthrough(self):
        import asyncio
        from pytans import aio

        data = make_text(50_000)

        async def run():
            packed = io.BytesIO()
            await aio.compress_stream(io.BytesIO(data), packed,
                                      block_size=16 * 1024, transform="lz77")
            packed.seek(0)
            restored = io.BytesIO()
            await aio.decompress_stream(packed, restored)
            return restored.getvalue()

        assert asyncio.run(run()) == data

    def test_cli_transform(self, tmp_path):
        from pytans.cli import main

        src = tmp_path / "data.txt"
        src.write_bytes(make_text(150_000))
        original = src.read_bytes()

        assert main(["compress", str(src), "--transform", "bwt"]) == 0
        packed = tmp_path / "data.txt.tans"
        plain_packed = tmp_path / "plain.tans"
        assert main(["compress", str(src), "-o", str(plain_packed)]) == 0
        assert packed.stat().st_size < plain_packed.stat().st_size

        src.unlink()
        assert main(["decompress", str(packed)]) == 0
        assert src.read_bytes() == original
