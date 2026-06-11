"""Optional modeling transforms: BWT and LZ77 front ends for tANS.

tANS alone is an order-0 entropy coder and cannot exploit repetition.
These transforms reshape the input so that repetition *becomes* symbol
skew the entropy stage can use:

- ``"bwt"`` — Burrows-Wheeler transform + move-to-front + zero run
  splitting: bzip2's pipeline. Best on text and structured data.
- ``"lz77"`` — greedy hash-chain match-finder emitting zstd-style
  substreams (literals, literal-run lengths, match lengths, offset
  buckets, raw offset extra bits): the zlib/zstd pipeline shape.

Each transform turns bytes into a list of substreams whose distributions
differ, so the frame layer compresses every substream with its own
fitted tANS table. Both directions are pure Python; the BWT suffix sort
uses numpy automatically when it is importable (orders of magnitude
faster), falling back to a pure-Python prefix-doubling sort.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from ._bitio import BitWriter, TailBitReader
from ._varint import read_uvarint, write_uvarint
from .exceptions import CorruptedDataError

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _np = None


# --------------------------------------------------------------------- BWT

def _sort_rotations_numpy(data: bytes) -> List[int]:
    n = len(data)
    rank = _np.frombuffer(data, dtype=_np.uint8).astype(_np.int64)
    k = 1
    while True:
        second = _np.roll(rank, -k)
        order = _np.lexsort((second, rank))
        key1, key2 = rank[order], second[order]
        changed = _np.ones(n, dtype=bool)
        changed[1:] = (key1[1:] != key1[:-1]) | (key2[1:] != key2[:-1])
        new_rank = _np.empty(n, dtype=_np.int64)
        new_rank[order] = _np.cumsum(changed) - 1
        rank = new_rank
        if rank[order[-1]] == n - 1 or k >= n:
            return order.tolist()
        k <<= 1


def _sort_rotations_pure(data: bytes) -> List[int]:
    n = len(data)
    rank = list(data)
    k = 1
    while True:
        order = sorted(range(n), key=lambda i: (rank[i], rank[(i + k) % n]))
        new_rank = [0] * n
        r = -1
        prev = None
        for i in order:
            key = (rank[i], rank[(i + k) % n])
            if key != prev:
                r += 1
                prev = key
            new_rank[i] = r
        rank = new_rank
        if r == n - 1 or k >= n:
            return order
        k <<= 1


def _bwt(data: bytes) -> Tuple[bytes, int]:
    """Cyclic-rotation BWT -> (last column, row index of the original)."""
    if len(data) < 2:
        return data, 0
    sorter = _sort_rotations_numpy if _np is not None else _sort_rotations_pure
    order = sorter(data)
    primary = order.index(0)
    last = bytes(data[i - 1] for i in order)
    return last, primary


def _ibwt(last: bytes, primary: int) -> bytes:
    n = len(last)
    if n < 2:
        return last
    if not 0 <= primary < n:
        raise CorruptedDataError("BWT primary index out of range")
    counts = [0] * 256
    for byte in last:
        counts[byte] += 1
    starts = [0] * 256
    total = 0
    for value in range(256):
        starts[value] = total
        total += counts[value]
    seen = [0] * 256
    lf = [0] * n
    for i, byte in enumerate(last):
        lf[i] = starts[byte] + seen[byte]
        seen[byte] += 1
    out = bytearray(n)
    row = primary
    for j in range(n - 1, -1, -1):
        out[j] = last[row]
        row = lf[row]
    return bytes(out)


def _mtf(data: bytes) -> bytes:
    table = list(range(256))
    out = bytearray()
    for byte in data:
        index = table.index(byte)
        out.append(index)
        if index:
            table.pop(index)
            table.insert(0, byte)
    return bytes(out)


def _imtf(data: bytes) -> bytes:
    table = list(range(256))
    out = bytearray()
    for index in data:
        byte = table[index]
        out.append(byte)
        if index:
            table.pop(index)
            table.insert(0, byte)
    return bytes(out)


def _zero_rle_split(data: bytes) -> Tuple[bytes, bytes]:
    """Collapse zero runs: one 0 in the symbol stream per run, lengths
    (minus one, 255-escaped) in a separate stream."""
    syms = bytearray()
    runs = bytearray()
    i = 0
    n = len(data)
    while i < n:
        byte = data[i]
        syms.append(byte)
        if byte == 0:
            j = i
            while j < n and data[j] == 0:
                j += 1
            run = j - i - 1
            while run >= 255:
                runs.append(255)
                run -= 255
            runs.append(run)
            i = j
        else:
            i += 1
    return bytes(syms), bytes(runs)


def _zero_rle_join(syms: bytes, runs: bytes) -> bytes:
    out = bytearray()
    ri = 0
    for byte in syms:
        if byte == 0:
            run = 0
            while True:
                if ri >= len(runs):
                    raise CorruptedDataError("zero-run stream exhausted")
                value = runs[ri]
                ri += 1
                run += value
                if value < 255:
                    break
            out += b"\x00" * (run + 1)
        else:
            out.append(byte)
    if ri != len(runs):
        raise CorruptedDataError("unused data in zero-run stream")
    return bytes(out)


def bwt_encode(data: bytes) -> List[bytes]:
    last, primary = _bwt(data)
    syms, runs = _zero_rle_split(_mtf(last))
    return [write_uvarint(primary) + syms, runs]


def bwt_decode(streams: List[bytes]) -> bytes:
    if len(streams) != 2:
        raise CorruptedDataError("BWT frame must carry 2 substreams")
    primary, pos = read_uvarint(streams[0], 0)
    last = _imtf(_zero_rle_join(streams[0][pos:], streams[1]))
    return _ibwt(last, primary)


# -------------------------------------------------------------------- LZ77

_MIN_MATCH = 4
_MAX_CHAIN = 48


def _byte_escape(values) -> bytes:
    """Non-negative ints as bytes; 255 means 'add 255 and continue'."""
    out = bytearray()
    for value in values:
        while value >= 255:
            out.append(255)
            value -= 255
        out.append(value)
    return bytes(out)


def _byte_unescape(data: bytes, count: int) -> List[int]:
    values = []
    pos = 0
    for _ in range(count):
        value = 0
        while True:
            if pos >= len(data):
                raise CorruptedDataError("escaped-length stream exhausted")
            byte = data[pos]
            pos += 1
            value += byte
            if byte < 255:
                break
        values.append(value)
    if pos != len(data):
        raise CorruptedDataError("unused data in escaped-length stream")
    return values


def _lz77_parse(data: bytes) -> Tuple[bytes, List[Tuple[int, int, int]]]:
    """Greedy parse -> (literals, [(literal_run, match_len, offset)]).

    The final sequence may have match_len 0 (trailing literals only).
    """
    n = len(data)
    table: Dict[bytes, List[int]] = {}

    def remember(pos: int) -> None:
        bucket = table.setdefault(data[pos:pos + _MIN_MATCH], [])
        bucket.append(pos)
        if len(bucket) > 2 * _MAX_CHAIN:
            del bucket[:_MAX_CHAIN]

    literals = bytearray()
    sequences: List[Tuple[int, int, int]] = []
    literal_run = 0
    i = 0
    while i < n:
        best_len = 0
        best_off = 0
        if i + _MIN_MATCH <= n:
            chain = table.get(data[i:i + _MIN_MATCH])
            if chain:
                for j in reversed(chain[-_MAX_CHAIN:]):
                    length = _MIN_MATCH
                    while i + length < n and data[j + length] == data[i + length]:
                        length += 1
                    if length > best_len:
                        best_len, best_off = length, i - j
                        if length >= 512:
                            break
        if best_len >= _MIN_MATCH:
            sequences.append((literal_run, best_len, best_off))
            literal_run = 0
            end = i + best_len
            step = 1 if best_len < 64 else 4
            for p in range(i, min(end, n - _MIN_MATCH), step):
                remember(p)
            i = end
        else:
            literals.append(data[i])
            literal_run += 1
            if i + _MIN_MATCH <= n:
                remember(i)
            i += 1
    if literal_run:
        sequences.append((literal_run, 0, 0))
    return bytes(literals), sequences


def lz77_encode(data: bytes) -> List[bytes]:
    literals, sequences = _lz77_parse(data)
    matches = [s for s in sequences if s[1]]

    offsets_writer = BitWriter()
    for _, _, offset in reversed(matches):
        # Reversed so the tail reader pops them back in forward order.
        offsets_writer.write(offset, max(offset.bit_length() - 1, 0))
    extras, extra_bits = offsets_writer.finish()

    return [
        write_uvarint(len(sequences)) + _byte_escape(s[0] for s in sequences),
        literals,
        _byte_escape(s[1] - _MIN_MATCH for s in matches),
        bytes(s[2].bit_length() for s in matches),
        write_uvarint(extra_bits) + extras,
    ]


def lz77_decode(streams: List[bytes]) -> bytes:
    if len(streams) != 5:
        raise CorruptedDataError("LZ77 frame must carry 5 substreams")
    runs_stream, literals, lens_stream, buckets, extras_stream = streams

    n_sequences, pos = read_uvarint(runs_stream, 0)
    literal_runs = _byte_unescape(runs_stream[pos:], n_sequences)
    n_matches = len(buckets)
    if not n_sequences - 1 <= n_matches <= n_sequences:
        raise CorruptedDataError("LZ77 sequence counts inconsistent")
    match_lens = _byte_unescape(lens_stream, n_matches)

    extra_bits, pos = read_uvarint(extras_stream, 0)
    extras = TailBitReader(extras_stream[pos:], extra_bits)

    out = bytearray()
    li = 0
    for s in range(n_sequences):
        run = literal_runs[s]
        if li + run > len(literals):
            raise CorruptedDataError("LZ77 literal stream exhausted")
        out += literals[li:li + run]
        li += run
        if s >= n_matches:
            continue
        bucket = buckets[s]
        if bucket < 1:
            raise CorruptedDataError("invalid LZ77 offset bucket")
        offset = (1 << (bucket - 1)) | extras.read(bucket - 1)
        if offset > len(out):
            raise CorruptedDataError("LZ77 offset reaches before stream start")
        for _ in range(match_lens[s] + _MIN_MATCH):  # byte-wise: overlap is legal
            out.append(out[-offset])
    if li != len(literals) or extras.bits_left:
        raise CorruptedDataError("unused data in LZ77 substreams")
    return bytes(out)


# ----------------------------------------------------------------- registry

_Encoder = Callable[[bytes], List[bytes]]
_Decoder = Callable[[List[bytes]], bytes]

#: name -> (wire id, encoder, decoder)
TRANSFORMS: Dict[str, Tuple[int, _Encoder, _Decoder]] = {
    "bwt": (1, bwt_encode, bwt_decode),
    "lz77": (2, lz77_encode, lz77_decode),
}

_BY_ID: Dict[int, Tuple[str, _Encoder, _Decoder]] = {
    wire_id: (name, encode, decode)
    for name, (wire_id, encode, decode) in TRANSFORMS.items()
}


def get_transform(name: str) -> Tuple[int, _Encoder, _Decoder]:
    try:
        return TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"unknown transform {name!r}; available: {sorted(TRANSFORMS)}"
        ) from None


def get_transform_by_id(wire_id: int) -> Tuple[str, _Encoder, _Decoder]:
    try:
        return _BY_ID[wire_id]
    except KeyError:
        raise CorruptedDataError(f"unknown transform id {wire_id}") from None
