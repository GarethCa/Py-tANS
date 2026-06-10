import random

import pytest

from pytans._bitio import BitWriter, TailBitReader
from pytans.exceptions import CorruptedDataError


def test_roundtrip_lifo_order():
    writer = BitWriter()
    writer.write(0b101, 3)
    writer.write(0b0110, 4)
    writer.write(0b1, 1)
    payload, bit_length = writer.finish()
    assert bit_length == 8

    reader = TailBitReader(payload, bit_length)
    assert reader.read(1) == 0b1
    assert reader.read(4) == 0b0110
    assert reader.read(3) == 0b101
    assert reader.bits_left == 0


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random_groups(seed):
    rng = random.Random(seed)
    groups = [(rng.randrange(1 << n), n) for n in (rng.randrange(16) for _ in range(500))]

    writer = BitWriter()
    for value, nbits in groups:
        writer.write(value, nbits)
    payload, bit_length = writer.finish()
    assert bit_length == sum(n for _, n in groups)
    assert len(payload) == (bit_length + 7) // 8

    reader = TailBitReader(payload, bit_length)
    for value, nbits in reversed(groups):
        assert reader.read(nbits) == value
    assert reader.bits_left == 0


def test_zero_bit_operations_are_noops():
    writer = BitWriter()
    writer.write(0xFFFF, 0)
    payload, bit_length = writer.finish()
    assert payload == b"" and bit_length == 0

    reader = TailBitReader(b"", 0)
    assert reader.read(0) == 0


def test_write_masks_excess_bits():
    writer = BitWriter()
    writer.write(0b111111, 2)  # only the low 2 bits should land
    payload, bit_length = writer.finish()
    reader = TailBitReader(payload, bit_length)
    assert reader.read(2) == 0b11


def test_reader_underflow_raises():
    writer = BitWriter()
    writer.write(0b101, 3)
    payload, bit_length = writer.finish()
    reader = TailBitReader(payload, bit_length)
    with pytest.raises(CorruptedDataError):
        reader.read(4)


def test_reader_rejects_inconsistent_bit_length():
    with pytest.raises(CorruptedDataError):
        TailBitReader(b"\x00", 9)
    with pytest.raises(CorruptedDataError):
        TailBitReader(b"\x00", -1)
