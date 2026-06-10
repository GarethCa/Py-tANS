"""Frequency normalisation and state-table construction for tANS.

The construction follows the Finite State Entropy (FSE) scheme used by
Zstandard: symbol counts are normalised so they sum to the table size
(a power of two), then symbols are spread across the state table with a
fixed coprime step.

Reference: J. Duda, "Asymmetric numeral systems" (arXiv:1311.2540) and
Y. Collet's FSE implementation (https://github.com/Cyan4973/FiniteStateEntropy).
"""

from __future__ import annotations

from typing import Dict, List, Mapping

#: Smallest supported table_log. Below 16 states the spread step is no
#: longer coprime with the table size and states would collide.
MIN_TABLE_LOG = 4

#: Largest supported table_log. The encoder packs the bit count into the
#: upper 16 bits of ``delta_nb_bits``, which bounds the state range.
MAX_TABLE_LOG = 15

#: Default upper bound used when choosing a table_log automatically.
DEFAULT_MAX_TABLE_LOG = 12


def _validate_table_log(table_log: int) -> None:
    if not isinstance(table_log, int) or not MIN_TABLE_LOG <= table_log <= MAX_TABLE_LOG:
        raise ValueError(
            f"table_log must be an int in [{MIN_TABLE_LOG}, {MAX_TABLE_LOG}], got {table_log!r}"
        )


def optimal_table_log(
    sample_size: int,
    alphabet_size: int,
    max_table_log: int = DEFAULT_MAX_TABLE_LOG,
) -> int:
    """Choose a table_log suited to ``sample_size`` symbols drawn from
    ``alphabet_size`` distinct values.

    Mirrors FSE's heuristic: small inputs get small tables (less header
    overhead), while the table always stays large enough to give every
    symbol a slot plus some precision headroom.
    """
    _validate_table_log(max_table_log)
    if alphabet_size < 1:
        raise ValueError("alphabet_size must be >= 1")
    if alphabet_size > (1 << max_table_log):
        raise ValueError(
            f"alphabet of {alphabet_size} symbols cannot fit in a "
            f"2**{max_table_log} state table"
        )
    log = max(sample_size - 1, 0).bit_length() - 2
    log = max(log, (alphabet_size - 1).bit_length() + 1, MIN_TABLE_LOG)
    log = min(log, max_table_log)
    if (1 << log) < alphabet_size:
        log = (alphabet_size - 1).bit_length()
    return log


def normalize_counts(counts: Mapping[int, int], table_log: int) -> Dict[int, int]:
    """Scale raw symbol counts so they sum to ``2**table_log``.

    Every symbol with a non-zero raw count keeps a normalised count of at
    least 1 (so it stays encodable). The result is keyed in ascending
    symbol order, which makes the table construction canonical: equal
    normalised counts always produce identical coding tables.
    """
    _validate_table_log(table_log)
    table_size = 1 << table_log

    for s, c in counts.items():
        if not isinstance(s, int) or not 0 <= s <= 255:
            raise ValueError(f"symbols must be byte values 0-255, got {s!r}")
        if c < 0:
            raise ValueError(f"negative count for symbol {s}")
    symbols = sorted(s for s, c in counts.items() if c > 0)
    if not symbols:
        raise ValueError("counts must contain at least one symbol with a positive count")
    if len(symbols) > table_size:
        raise ValueError(
            f"{len(symbols)} symbols cannot fit in a table of {table_size} states; "
            f"increase table_log"
        )
    if len(symbols) == 1:
        return {symbols[0]: table_size}

    total = sum(counts[s] for s in symbols)
    norm = {s: max(1, (counts[s] * table_size) // total) for s in symbols}

    diff = table_size - sum(norm.values())
    if diff > 0:
        # Hand all the shortfall to the most frequent symbol; it suffers
        # the least relative distortion.
        norm[max(symbols, key=lambda s: (counts[s], -s))] += diff
    while diff < 0:
        # Rounding up the rare symbols overshot the table: claw back the
        # excess from the largest normalised counts.
        s = max(symbols, key=lambda s: (norm[s], -s))
        take = min(-diff, norm[s] - 1)
        norm[s] -= take
        diff += take
    return norm


def spread_symbols(normalized_counts: Mapping[int, int], table_log: int) -> List[int]:
    """Distribute symbols across the state table.

    Uses FSE's spread: stepping through the table by an odd constant
    (coprime with the power-of-two table size) visits every slot exactly
    once, scattering each symbol's states roughly uniformly.
    """
    _validate_table_log(table_log)
    table_size = 1 << table_log
    if sum(normalized_counts.values()) != table_size:
        raise ValueError("normalized counts must sum to the table size")
    step = (table_size >> 1) + (table_size >> 3) + 3
    mask = table_size - 1
    state_table: List[int] = [0] * table_size
    pos = 0
    for symbol, occurrences in normalized_counts.items():
        for _ in range(occurrences):
            state_table[pos] = symbol
            pos = (pos + step) & mask
    assert pos == 0, "spread step did not cycle the whole table"
    return state_table
