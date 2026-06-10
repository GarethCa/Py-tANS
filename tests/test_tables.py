from collections import Counter

import pytest

from pytans.tables import (
    MAX_TABLE_LOG,
    MIN_TABLE_LOG,
    normalize_counts,
    optimal_table_log,
    spread_symbols,
)


class TestNormalizeCounts:
    def test_sums_to_table_size_with_min_one(self):
        counts = {0: 1000, 1: 100, 2: 10, 3: 1}
        for table_log in range(MIN_TABLE_LOG, MAX_TABLE_LOG + 1):
            norm = normalize_counts(counts, table_log)
            assert sum(norm.values()) == 1 << table_log
            assert all(c >= 1 for c in norm.values())
            assert set(norm) == set(counts)

    def test_already_normalized_is_identity(self):
        counts = {10: 10, 11: 10, 12: 12}  # sums to 32
        assert normalize_counts(counts, 5) == counts

    def test_result_is_keyed_in_ascending_symbol_order(self):
        norm = normalize_counts({200: 5, 3: 5, 100: 6}, 5)
        assert list(norm) == [3, 100, 200]

    def test_single_symbol_takes_whole_table(self):
        assert normalize_counts({65: 123}, 6) == {65: 64}

    def test_zero_counts_are_dropped(self):
        norm = normalize_counts({0: 50, 1: 0, 2: 50}, 5)
        assert set(norm) == {0, 2}

    def test_preserves_rare_symbols(self):
        # Symbol 3 is ~0.01% of the data but must keep a slot.
        norm = normalize_counts({0: 10000, 3: 1}, 5)
        assert norm[3] >= 1
        assert sum(norm.values()) == 32

    def test_errors(self):
        with pytest.raises(ValueError):
            normalize_counts({}, 5)
        with pytest.raises(ValueError):
            normalize_counts({0: 0}, 5)
        with pytest.raises(ValueError):
            normalize_counts({0: -1, 1: 5}, 5)
        with pytest.raises(ValueError):
            normalize_counts({256: 5}, 5)  # not a byte value
        with pytest.raises(ValueError):
            normalize_counts({0: 1}, MAX_TABLE_LOG + 1)
        with pytest.raises(ValueError):
            normalize_counts({0: 1}, MIN_TABLE_LOG - 1)
        with pytest.raises(ValueError):
            # 17 symbols cannot fit 16 states
            normalize_counts({i: 1 for i in range(17)}, 4)


class TestSpreadSymbols:
    def test_each_symbol_appears_exactly_count_times(self):
        norm = normalize_counts({0: 10, 1: 10, 2: 12}, 5)
        table = spread_symbols(norm, 5)
        assert len(table) == 32
        assert Counter(table) == norm

    def test_full_alphabet(self):
        norm = {i: 1 for i in range(256)}
        table = spread_symbols(norm, 8)
        assert Counter(table) == norm

    def test_rejects_unnormalized_counts(self):
        with pytest.raises(ValueError):
            spread_symbols({0: 3}, 5)


class TestOptimalTableLog:
    def test_within_bounds(self):
        for size in (1, 10, 1000, 10**9):
            for alphabet in (1, 2, 200, 256):
                log = optimal_table_log(size, alphabet)
                assert MIN_TABLE_LOG <= log <= MAX_TABLE_LOG
                assert (1 << log) >= alphabet

    def test_small_inputs_get_small_tables(self):
        assert optimal_table_log(20, 3) < optimal_table_log(10**6, 3)

    def test_respects_max(self):
        assert optimal_table_log(10**6, 2, max_table_log=6) == 6

    def test_rejects_oversized_alphabet(self):
        with pytest.raises(ValueError):
            optimal_table_log(100, 257, max_table_log=8)
        with pytest.raises(ValueError):
            optimal_table_log(100, 0)
