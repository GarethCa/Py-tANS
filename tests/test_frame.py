import random

import pytest

from pytans import CorruptedDataError, compress, decompress


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"ab",
        b"1102010120",
        b"a" * 100_000,
        b"the quick brown fox jumps over the lazy dog " * 200,
        bytes(range(256)) * 10,
        bytes([0, 255] * 5_000),
    ],
    ids=["empty", "one-byte", "two-bytes", "notebook", "runs", "text", "all-bytes", "binary"],
)
def test_roundtrip(data):
    assert decompress(compress(data)) == data


def test_roundtrip_random_bytes():
    rng = random.Random(7)
    data = bytes(rng.randrange(256) for _ in range(20_000))
    blob = compress(data)
    assert decompress(blob) == data
    # Incompressible input falls back to raw storage with tiny overhead.
    assert len(blob) <= len(data) + 16


def test_compresses_skewed_data():
    data = b"abracadabra alakazam " * 500
    blob = compress(data)
    assert len(blob) < len(data) // 2
    assert decompress(blob) == data


def test_explicit_table_log_roundtrips():
    data = b"mississippi" * 100
    for table_log in (4, 8, 15):
        assert decompress(compress(data, table_log=table_log)) == data


def test_accepts_bytearray_and_memoryview():
    data = b"hello hello hello"
    assert decompress(compress(bytearray(data))) == data
    assert decompress(memoryview(compress(data))) == data


class TestCorruptFrames:
    def test_not_a_frame(self):
        with pytest.raises(CorruptedDataError):
            decompress(b"")
        with pytest.raises(CorruptedDataError):
            decompress(b"PK\x03\x04 not tans data")

    def test_bad_version(self):
        blob = bytearray(compress(b"hello hello"))
        blob[4] = 99
        with pytest.raises(CorruptedDataError):
            decompress(bytes(blob))

    def test_bad_mode(self):
        blob = bytearray(compress(b"hello hello"))
        blob[5] = 7
        with pytest.raises(CorruptedDataError):
            decompress(bytes(blob))

    def test_truncated_frame(self):
        blob = compress(b"compressible compressible compressible")
        for cut in (3, 6, 10, len(blob) - 1):
            with pytest.raises(CorruptedDataError):
                decompress(blob[:cut])

    def test_trailing_garbage(self):
        blob = compress(b"hello hello hello")
        with pytest.raises(CorruptedDataError):
            decompress(blob + b"x")

    def test_payload_corruption_does_not_pass_silently(self):
        data = b"sphinx of black quartz judge my vow " * 300
        blob = bytearray(compress(data))
        rng = random.Random(0)
        detected = 0
        trials = 20
        for _ in range(trials):
            corrupted = bytearray(blob)
            # Skip the magic (caught trivially) and the final byte, whose
            # zero-pad bits are legitimately ignored by the decoder.
            corrupted[rng.randrange(8, len(blob) - 1)] ^= 1 << rng.randrange(8)
            try:
                result = decompress(bytes(corrupted))
            except CorruptedDataError:
                detected += 1
            else:
                if result != data:
                    detected += 1
        # tANS has no checksum, but the final-state and bit-accounting
        # checks must stop corruption from round-tripping unnoticed.
        assert detected == trials
