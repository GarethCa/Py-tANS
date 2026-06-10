import math
import random
from collections import Counter

import pytest

from pytans import CorruptedDataError, TansCoder


def make_data(spec, n, seed=0):
    """Random bytes drawn from {symbol: weight} ``spec``."""
    rng = random.Random(seed)
    symbols = list(spec)
    weights = [spec[s] for s in symbols]
    return bytes(rng.choices(symbols, weights=weights, k=n))


def roundtrip(data, table_log=None):
    coder = TansCoder.from_data(data, table_log)
    payload, bit_length = coder.encode(data)
    assert coder.decode(payload, bit_length, len(data)) == data
    return payload, bit_length


class TestRoundtrip:
    def test_notebook_example(self):
        # The original notebook's worked example: alphabet {0,1,2} with
        # counts {10, 10, 12} on a 32-state table.
        data = b"1102010120"
        coder = TansCoder({ord("0"): 10, ord("1"): 10, ord("2"): 12}, table_log=5)
        payload, bit_length = coder.encode(data)
        assert coder.decode(payload, bit_length, len(data)) == data

    def test_empty_input(self):
        coder = TansCoder({0: 1, 1: 1})
        payload, bit_length = coder.encode(b"")
        assert coder.decode(payload, bit_length, 0) == b""

    def test_single_byte(self):
        roundtrip(b"x")

    def test_single_symbol_alphabet(self):
        payload, bit_length = roundtrip(b"a" * 10_000)
        # A one-symbol alphabet needs ~0 bits per symbol.
        assert len(payload) <= 4

    def test_all_256_byte_values(self):
        roundtrip(bytes(range(256)) * 7)

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize(
        "spec",
        [
            {97: 1, 98: 1},
            {97: 90, 98: 9, 99: 1},
            {i: i + 1 for i in range(64)},
        ],
    )
    def test_random_distributions(self, spec, seed):
        roundtrip(make_data(spec, 5_000, seed))

    @pytest.mark.parametrize("table_log", [4, 5, 8, 12, 15])
    def test_all_table_logs(self, table_log):
        roundtrip(make_data({97: 70, 98: 20, 99: 10}, 2_000), table_log)

    def test_random_bytes(self):
        rng = random.Random(42)
        roundtrip(bytes(rng.randrange(256) for _ in range(10_000)))


class TestCoderReuse:
    def test_shared_table_across_messages(self):
        # Dictionary-style usage: build the table from a sample, then
        # code other messages over the same alphabet.
        sample = make_data({101: 60, 116: 30, 32: 10}, 10_000, seed=1)
        encoder = TansCoder.from_data(sample)
        decoder = TansCoder(encoder.normalized_counts, encoder.table_log)
        for seed in range(5):
            message = make_data({101: 50, 116: 40, 32: 10}, 500, seed)
            payload, bit_length = encoder.encode(message)
            assert decoder.decode(payload, bit_length, len(message)) == message

    def test_tables_are_deterministic(self):
        a = TansCoder({5: 3, 9: 9, 200: 1}, table_log=7)
        b = TansCoder({200: 1, 9: 9, 5: 3}, table_log=7)  # different key order
        data = bytes([5, 9, 9, 200, 9, 5])
        assert a.encode(data) == b.encode(data)


class TestCompressionQuality:
    def test_close_to_shannon_entropy(self):
        spec = {97: 70, 98: 20, 99: 9, 100: 1}
        n = 50_000
        data = make_data(spec, n, seed=3)
        _, bit_length = roundtrip(data, table_log=12)

        counts = Counter(data)
        entropy = -sum(c / n * math.log2(c / n) for c in counts.values())
        # Within 2% of the entropy bound (plus the final-state flush).
        assert bit_length <= n * entropy * 1.02 + 64


class TestErrors:
    def test_unknown_symbol(self):
        coder = TansCoder({97: 1, 98: 1})
        with pytest.raises(ValueError, match="alphabet"):
            coder.encode(b"abc")

    def test_truncated_payload_detected(self):
        data = make_data({97: 3, 98: 1}, 1_000)
        coder = TansCoder.from_data(data)
        payload, bit_length = coder.encode(data)
        with pytest.raises(CorruptedDataError):
            coder.decode(payload[:-1], bit_length, len(data))

    def test_wrong_length_detected(self):
        data = make_data({97: 3, 98: 1}, 1_000)
        coder = TansCoder.from_data(data)
        payload, bit_length = coder.encode(data)
        for wrong in (len(data) - 1, len(data) + 1):
            with pytest.raises(CorruptedDataError):
                coder.decode(payload, bit_length, wrong)

    def test_empty_sample_rejected(self):
        with pytest.raises(ValueError):
            TansCoder.from_data(b"")

    def test_negative_length_rejected(self):
        coder = TansCoder({97: 1, 98: 1})
        with pytest.raises(ValueError):
            coder.decode(b"\x00\x00", 16, -1)
