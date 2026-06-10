"""The tANS encoder/decoder.

A :class:`TansCoder` is built once from symbol statistics and can then
encode and decode any number of byte strings drawn from that alphabet.
Tables are canonical: the same normalised counts and table_log always
produce the same coder, so an encoder and decoder built independently
from the same statistics interoperate.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Optional, Tuple, Union

from ._bitio import BitWriter, TailBitReader
from .exceptions import CorruptedDataError
from .tables import normalize_counts, optimal_table_log, spread_symbols

BytesLike = Union[bytes, bytearray, memoryview]


def _high_bit(value: int) -> int:
    """Position of the highest set bit (0 for values <= 1)."""
    return value.bit_length() - 1 if value > 1 else 0


class TansCoder:
    """A tabled ANS coder over byte symbols (values 0-255).

    Parameters
    ----------
    counts:
        Raw (or already normalised) occurrence counts per byte value.
        They are normalised internally to sum to ``2**table_log``.
    table_log:
        Base-2 log of the state table size. ``None`` picks a sensible
        size from the counts.
    """

    def __init__(self, counts: Mapping[int, int], table_log: Optional[int] = None) -> None:
        if table_log is None:
            present = [s for s, c in counts.items() if c > 0]
            table_log = optimal_table_log(sum(counts.values()), max(len(present), 1))
        self.table_log = table_log
        self.table_size = 1 << table_log
        self._counts = normalize_counts(counts, table_log)

        state_table = spread_symbols(self._counts, table_log)

        # Encoder tables. For each symbol: how many bits to flush for a
        # given state (packed FSE-style into the upper 16 bits), and where
        # its block of next-states starts in the coding table.
        self._delta_nb_bits: List[Optional[int]] = [None] * 256
        self._delta_find_state: List[int] = [0] * 256
        total = 0
        for symbol, occurrences in self._counts.items():
            max_bits_out = table_log - _high_bit(occurrences - 1)
            self._delta_nb_bits[symbol] = (max_bits_out << 16) - (occurrences << max_bits_out)
            self._delta_find_state[symbol] = total - occurrences
            total += occurrences

        # coding_table[cumulative rank of state's symbol] = next state.
        cumulative: Dict[int, int] = {}
        running = 0
        for symbol, occurrences in self._counts.items():
            cumulative[symbol] = running
            running += occurrences
        self._coding_table = [0] * self.table_size
        for i, symbol in enumerate(state_table):
            self._coding_table[cumulative[symbol]] = self.table_size + i
            cumulative[symbol] += 1

        # Decoder tables, indexed by state.
        next_rank = dict(self._counts)
        self._decode_symbol = bytearray(self.table_size)
        self._decode_nb_bits = [0] * self.table_size
        self._decode_new_state = [0] * self.table_size
        for i, symbol in enumerate(state_table):
            rank = next_rank[symbol]
            next_rank[symbol] = rank + 1
            nb_bits = table_log - _high_bit(rank)
            self._decode_symbol[i] = symbol
            self._decode_nb_bits[i] = nb_bits
            self._decode_new_state[i] = (rank << nb_bits) - self.table_size

    @classmethod
    def from_data(cls, sample: BytesLike, table_log: Optional[int] = None) -> "TansCoder":
        """Build a coder from the byte statistics of ``sample``."""
        sample = bytes(sample)
        if not sample:
            raise ValueError("cannot build a coder from an empty sample")
        return cls(Counter(sample), table_log)

    @property
    def normalized_counts(self) -> Dict[int, int]:
        """The normalised per-symbol counts (sum to ``table_size``)."""
        return dict(self._counts)

    @property
    def symbols(self) -> List[int]:
        """Byte values this coder can encode, in ascending order."""
        return list(self._counts)

    def encode(self, data: BytesLike) -> Tuple[bytes, int]:
        """Encode ``data`` and return ``(payload, bit_length)``.

        ``bit_length`` counts the valid bits in ``payload`` (the final
        byte is zero-padded); both it and ``len(data)`` are needed to
        decode, so store them alongside the payload (as
        :func:`pytans.compress` does).
        """
        writer = BitWriter()
        state = self.table_size
        delta_nb_bits = self._delta_nb_bits
        delta_find_state = self._delta_find_state
        coding_table = self._coding_table
        for byte in bytes(data):
            packed = delta_nb_bits[byte]
            if packed is None:
                raise ValueError(f"symbol {byte} does not appear in this coder's alphabet")
            nb_bits = (state + packed) >> 16
            writer.write(state, nb_bits)
            state = coding_table[(state >> nb_bits) + delta_find_state[byte]]
        writer.write(state - self.table_size, self.table_log)
        return writer.finish()

    def decode(self, payload: BytesLike, bit_length: int, length: int) -> bytes:
        """Decode ``length`` symbols from ``payload``.

        ``bit_length`` and ``length`` must be the values produced by
        :meth:`encode`. Raises :class:`CorruptedDataError` if the stream
        does not validate.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if bit_length < self.table_log:
            raise CorruptedDataError("payload too short to contain a final state")
        reader = TailBitReader(bytes(payload), bit_length)
        state = reader.read(self.table_log)
        decode_symbol = self._decode_symbol
        decode_nb_bits = self._decode_nb_bits
        decode_new_state = self._decode_new_state
        out = bytearray(length)
        # The decoder walks the stream backwards, so symbols come out in
        # reverse encode order: fill the output back to front.
        for i in range(length - 1, -1, -1):
            out[i] = decode_symbol[state]
            state = decode_new_state[state] + reader.read(decode_nb_bits[state])
        if state != 0 or reader.bits_left != 0:
            raise CorruptedDataError("bitstream failed integrity check after decoding")
        return bytes(out)
