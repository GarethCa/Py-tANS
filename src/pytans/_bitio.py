"""Bit-level I/O for the tANS coder.

tANS is a LIFO code: the encoder appends groups of bits to the tail of the
stream and the decoder consumes them from the tail backwards. The writer
packs bits MSB-first into a bytearray; the reader pops groups of bits off
the end given the total number of valid bits.
"""

from __future__ import annotations

from .exceptions import CorruptedDataError


class BitWriter:
    """Appends bit groups to a growing buffer, MSB-first."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def write(self, value: int, nbits: int) -> None:
        """Append the low ``nbits`` bits of ``value``."""
        if nbits == 0:
            return
        self._acc = (self._acc << nbits) | (value & ((1 << nbits) - 1))
        self._nbits += nbits
        while self._nbits >= 8:
            self._nbits -= 8
            self._buf.append((self._acc >> self._nbits) & 0xFF)
        self._acc &= (1 << self._nbits) - 1

    def finish(self) -> "tuple[bytes, int]":
        """Zero-pad to a whole byte and return ``(payload, bit_length)``."""
        bit_length = len(self._buf) * 8 + self._nbits
        if self._nbits:
            self._buf.append((self._acc << (8 - self._nbits)) & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._buf), bit_length


class TailBitReader:
    """Pops bit groups off the tail of a buffer written by :class:`BitWriter`."""

    def __init__(self, data: bytes, bit_length: int) -> None:
        if bit_length < 0 or bit_length > len(data) * 8:
            raise CorruptedDataError("bit length inconsistent with payload size")
        self._data = data
        self.bits_left = bit_length

    def read(self, nbits: int) -> int:
        """Remove and return the last ``nbits`` bits of the remaining stream."""
        if nbits == 0:
            return 0
        start = self.bits_left - nbits
        if start < 0:
            raise CorruptedDataError("bitstream underflow")
        end = self.bits_left
        self.bits_left = start
        first_byte = start >> 3
        last_byte = (end - 1) >> 3
        chunk = int.from_bytes(self._data[first_byte : last_byte + 1], "big")
        shift = (last_byte + 1) * 8 - end
        return (chunk >> shift) & ((1 << nbits) - 1)
