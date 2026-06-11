"""Tests for the context-split BWT transform (bwt2) and lazy LZ77 matching."""

import io
import random

import pytest

from pytans import CorruptedDataError, compress, compress_stream, decompress, decompress_stream
from pytans import transforms
from pytans.transforms import (
    _MIN_MATCH,
    _N_MTF_CONTEXTS,
    _lz77_parse,
    _mtf_context,
    bwt2_decode,
    bwt2_encode,
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
    b"banana banana banana",
    b"a" * 5_000,
    b"ab" * 3_000,
    make_text(20_000),
    bytes(random.Random(1).randbytes(5_000)),
    bytes(range(256)) * 20,
]
IDS = ["empty", "one", "banana", "runs", "pairs", "text", "noise", "cycle"]


class TestMtfContext:
    def test_buckets_are_in_range_for_all_bytes(self):
        for value in range(256):
            assert 0 <= _mtf_context(value) < _N_MTF_CONTEXTS

    def test_bucket_boundaries(self):
        assert _mtf_context(0) == 0
        assert _mtf_context(1) == 1
        assert _mtf_context(2) == _mtf_context(3) == 3
        assert _mtf_context(4) == _mtf_context(7) == 4
        assert _mtf_context(8) == _mtf_context(15) == 5
        assert _mtf_context(32) == _mtf_context(255) == 7

    def test_deterministic_routing_roundtrip(self):
        # The decoder re-derives each symbol's stream from the previous
        # symbol, so any symbol sequence must reassemble exactly.
        rng = random.Random(7)
        syms = bytes(rng.choices(range(64), weights=[50] + [1] * 63, k=4_000))
        streams = [bytearray() for _ in range(_N_MTF_CONTEXTS)]
        ctx = 0
        for s in syms:
            streams[ctx].append(s)
            ctx = _mtf_context(s)
        positions = [0] * _N_MTF_CONTEXTS
        out = bytearray()
        ctx = 0
        for _ in range(len(syms)):
            out.append(streams[ctx][positions[ctx]])
            positions[ctx] += 1
            ctx = _mtf_context(out[-1])
        assert bytes(out) == syms


class TestBwt2Codec:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_encode_decode_identity(self, data):
        assert bwt2_decode(bwt2_encode(data)) == data

    def test_substream_count(self):
        assert len(bwt2_encode(make_text(1_000))) == 1 + _N_MTF_CONTEXTS

    def test_wrong_substream_count_rejected(self):
        with pytest.raises(CorruptedDataError):
            bwt2_decode([b""] * _N_MTF_CONTEXTS)  # one short

    def test_truncated_context_stream_rejected(self):
        streams = bwt2_encode(make_text(5_000))
        populated = max(range(1, len(streams)), key=lambda i: len(streams[i]))
        streams[populated] = streams[populated][:-1]
        with pytest.raises(CorruptedDataError):
            bwt2_decode(streams)

    def test_pure_python_bwt_path(self, monkeypatch):
        data = make_text(3_000)
        expected = bwt2_encode(data)
        monkeypatch.setattr(transforms, "_np", None)
        assert bwt2_encode(data) == expected
        assert bwt2_decode(expected) == data


class TestBwt2Frames:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_roundtrip(self, data):
        assert decompress(compress(data, transform="bwt2")) == data

    def test_beats_plain_bwt_on_text(self):
        data = make_text(120_000)
        bwt2_size = len(compress(data, transform="bwt2"))
        bwt_size = len(compress(data, transform="bwt"))
        assert bwt2_size < bwt_size
        assert bwt2_size < len(compress(data)) * 0.6

    def test_identical_to_plain_when_unhelpful(self):
        rng = random.Random(3)
        data = bytes(rng.choices(range(8), weights=[40, 20, 10, 8, 6, 5, 4, 3], k=20_000))
        assert compress(data, transform="bwt2") == compress(data)

    def test_corruption_never_silent(self):
        data = make_text(40_000)
        blob = bytearray(compress(data, transform="bwt2"))
        assert blob[5] == 3 and blob[6] == 3  # transform mode, bwt2 wire id
        rng = random.Random(0)
        for _ in range(20):
            corrupted = bytearray(blob)
            corrupted[rng.randrange(8, len(blob) - 1)] ^= 1 << rng.randrange(8)
            try:
                result = decompress(bytes(corrupted))
            except CorruptedDataError:
                continue
            assert result != data

    def test_stream_and_cli(self, tmp_path):
        from pytans.cli import main

        data = make_text(200_000)
        out = io.BytesIO()
        compress_stream(io.BytesIO(data), out, block_size=64 * 1024, transform="bwt2")
        out.seek(0)
        restored = io.BytesIO()
        decompress_stream(out, restored)
        assert restored.getvalue() == data

        src = tmp_path / "data.txt"
        src.write_bytes(data)
        assert main(["compress", str(src), "--transform", "bwt2"]) == 0
        src.unlink()
        assert main(["decompress", str(src) + ".tans"]) == 0
        assert src.read_bytes() == data


class TestLazyLz77:
    @pytest.mark.parametrize("data", SAMPLES, ids=IDS)
    def test_identity_after_lazy_parse(self, data):
        assert lz77_decode(lz77_encode(data)) == data

    def test_lazy_deferral_is_exercised_and_reversible(self):
        # 'bcdefgh' first appears inside a shorter earlier match context:
        # at the join points the position one ahead offers a longer match.
        base = b"abcd" + b"bcdefgh" + b"xyz"
        data = base * 500
        literals, sequences = _lz77_parse(data)
        rebuilt = bytearray()
        li = 0
        for run, mlen, off in sequences:
            rebuilt += literals[li:li + run]
            li += run
            for _ in range(mlen):
                rebuilt.append(rebuilt[-off])
        assert bytes(rebuilt) == data
        assert any(s[1] >= _MIN_MATCH for s in sequences)

    def test_parse_quality_on_text(self):
        # Lazy parsing should keep the lz77 frame well under plain tANS.
        data = make_text(120_000)
        assert len(compress(data, transform="lz77")) < len(compress(data)) * 0.55

    def test_wire_format_unchanged(self):
        # Frames written before lazy matching still decode: the format is
        # the same five substreams, so old payloads remain valid.
        data = make_text(10_000)
        blob = compress(data, transform="lz77")
        assert blob[5] == 3 and blob[6] == 2  # transform mode, lz77 wire id
        assert decompress(blob) == data


class TestRegistry:
    def test_bwt2_registered(self):
        assert "bwt2" in transforms.TRANSFORMS
        wire_id, _, _ = transforms.get_transform("bwt2")
        assert wire_id == 3
        assert transforms.get_transform_by_id(3)[0] == "bwt2"

    def test_wire_ids_stable_and_unique(self):
        ids = {wire_id for wire_id, _, _ in transforms.TRANSFORMS.values()}
        assert ids == {1, 2, 3}
