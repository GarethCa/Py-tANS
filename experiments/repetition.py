#!/usr/bin/env python3
"""Experiments: modeling front ends for tANS on repetitive data.

pytans alone is an order-0 entropy coder, so it cannot exploit
repetition. These experiments bolt classic modeling stages in front of
it and measure how much of the gap to zlib/bz2/lzma each one closes:

  mtf+tans       move-to-front, then tANS               (locality)
  bwt+tans       Burrows-Wheeler + MTF + zero-RLE + tANS (bzip2's pipeline)
  lz77+tans      greedy hash-chain match-finder + tANS   (zlib/zstd's pipeline)

LZ77 output is split zstd-style into separate streams (literals,
literal-run lengths, match lengths, offset buckets) so each gets a tANS
table fitted to its own distribution; offset extra bits are stored raw.

The LZ77 parse is verified by reconstruction. BWT/MTF/RLE are standard
invertible transforms (inverse not timed here). Sizes include all
streams plus per-stream frame headers.

Usage:  python experiments/repetition.py [--size BYTES]
"""

import argparse
import json
import math
import random
import time
import zlib
import bz2
import lzma
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

from pytans import compress

# --------------------------------------------------------------- corpora

def corpora(size):
    rng = random.Random(0)
    out = {}
    for name, path in (("English text", "/tmp/pytans-bench-wnp.txt"),
                       ("Python source", "/tmp/pysrc.txt"),
                       ("Sorted dictionary", "/usr/share/dict/words")):
        p = Path(path)
        if p.exists():
            out[name] = p.read_bytes()[:size]
    lines = []
    for i in range(60_000):
        lines.append(json.dumps({
            "ts": 1_750_000_000 + i,
            "level": rng.choice(["INFO"] * 8 + ["WARN", "ERROR"]),
            "service": rng.choice(["auth", "billing", "api"]),
            "latency_ms": rng.randrange(1, 500),
            "ok": rng.random() < 0.95,
            "request_id": f"{rng.randrange(16**12):012x}",
        }))
    out["JSON logs"] = "\n".join(lines).encode()[:size]
    out["Skewed, no repeats"] = bytes(
        rng.choices(range(8), weights=[40, 20, 10, 8, 6, 5, 4, 3], k=size))
    return out

# ------------------------------------------------------------ transforms

def mtf(data):
    table = list(range(256))
    out = bytearray()
    for b in data:
        i = table.index(b)
        out.append(i)
        if i:
            table.pop(i)
            table.insert(0, b)
    return bytes(out)


def bwt(data):
    """Suffix-array BWT of cyclic rotations via numpy prefix doubling."""
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    rank = arr.astype(np.int64)
    k = 1
    while True:
        second = np.roll(rank, -k)
        order = np.lexsort((second, rank))
        k1, k2 = rank[order], second[order]
        changed = np.ones(n, dtype=bool)
        changed[1:] = (k1[1:] != k1[:-1]) | (k2[1:] != k2[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[order] = np.cumsum(changed) - 1
        rank = new_rank
        if rank[order[-1]] == n - 1 or k >= n:
            return arr[(order - 1) % n].tobytes()
        k <<= 1


def zero_rle(data):
    """Split MTF output into (symbols-with-runs-collapsed, run lengths)."""
    syms = bytearray()
    runs = bytearray()
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        syms.append(b)
        if b == 0:
            j = i
            while j < n and data[j] == 0:
                j += 1
            run = j - i - 1
            while run >= 255:        # 255-escape continuation
                runs.append(255)
                run -= 255
            runs.append(run)
            i = j
        else:
            i += 1
    return bytes(syms), bytes(runs)

# ------------------------------------------------------------------ lz77

MIN_MATCH = 4
MAX_CHAIN = 48


def lz77_parse(data):
    """Greedy hash-chain parse -> (literals, [(litrun, mlen, offset)])."""
    n = len(data)
    table = {}
    lits = bytearray()
    seqs = []
    litrun = 0
    i = 0
    while i < n:
        best_len = 0
        best_off = 0
        if i + MIN_MATCH <= n:
            key = bytes(data[i:i + MIN_MATCH])
            chain = table.get(key)
            if chain:
                for j in reversed(chain[-MAX_CHAIN:]):
                    length = MIN_MATCH
                    while i + length < n and data[j + length] == data[i + length]:
                        length += 1
                    if length > best_len:
                        best_len, best_off = length, i - j
                        if length >= 512:
                            break
        if best_len >= MIN_MATCH:
            seqs.append((litrun, best_len, best_off))
            litrun = 0
            end = i + best_len
            step = 1 if best_len < 64 else 4
            for p in range(i, min(end, n - MIN_MATCH), step):
                bucket = table.setdefault(bytes(data[p:p + MIN_MATCH]), [])
                bucket.append(p)
                if len(bucket) > 2 * MAX_CHAIN:
                    del bucket[:MAX_CHAIN]
            i = end
        else:
            lits.append(data[i])
            litrun += 1
            if i + MIN_MATCH <= n:
                bucket = table.setdefault(bytes(data[i:i + MIN_MATCH]), [])
                bucket.append(i)
                if len(bucket) > 2 * MAX_CHAIN:
                    del bucket[:MAX_CHAIN]
            i += 1
    if litrun:
        seqs.append((litrun, 0, 0))
    return bytes(lits), seqs


def lz77_rebuild(lits, seqs):
    out = bytearray()
    li = 0
    for litrun, mlen, off in seqs:
        out += lits[li:li + litrun]
        li += litrun
        for _ in range(mlen):                 # byte-wise: handles overlap
            out.append(out[-off])
    return bytes(out)


def byte_escape(values):
    """Encode ints >= 0 as bytes with 255-escape continuation."""
    out = bytearray()
    for v in values:
        while v >= 255:
            out.append(255)
            v -= 255
        out.append(v)
    return bytes(out)


def lz77_size(data):
    lits, seqs = lz77_parse(data)
    assert lz77_rebuild(lits, seqs) == bytes(data), "parse not reversible"
    litruns = byte_escape(s[0] for s in seqs)
    mlens = byte_escape(s[1] - MIN_MATCH for s in seqs if s[1])
    offsets = [s[2] for s in seqs if s[1]]
    obuckets = bytes(o.bit_length() for o in offsets)
    extra_bits = sum(max(o.bit_length() - 1, 0) for o in offsets)
    streams = [lits, litruns, mlens, obuckets]
    return sum(len(compress(s)) for s in streams) + (extra_bits + 7) // 8

# ------------------------------------------------------------- pipelines

def order0_floor(data):
    n = len(data)
    counts = Counter(data)
    return -sum(c / n * math.log2(c / n) for c in counts.values()) * n / 8


def bwt_size(data, block=1 << 18):
    total = 0
    for start in range(0, len(data), block):
        chunk = data[start:start + block]
        syms, runs = zero_rle(mtf(bwt(chunk)))
        total += len(compress(syms)) + len(compress(runs))
    return total


def mtf_size(data):
    syms, runs = zero_rle(mtf(data))
    return len(compress(syms)) + len(compress(runs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1 << 20)
    args = parser.parse_args()

    pipelines = [
        ("tans", lambda d: len(compress(d))),
        ("mtf+tans", mtf_size),
        ("bwt+tans", bwt_size),
        ("lz77+tans", lz77_size),
        ("zlib-9", lambda d: len(zlib.compress(d, 9))),
        ("bz2-9", lambda d: len(bz2.compress(d, 9))),
        ("lzma", lambda d: len(lzma.compress(d))),
    ]
    names = [p[0] for p in pipelines]
    print(f"{'corpus':20s} {'floor':>7s} " + " ".join(f"{n:>9s}" for n in names))
    for cname, data in corpora(args.size).items():
        row = [f"{order0_floor(data) / len(data):>7.1%}"]
        for pname, fn in pipelines:
            t = time.perf_counter()
            size = fn(data)
            secs = time.perf_counter() - t
            row.append(f"{size / len(data):>9.1%}")
        print(f"{cname:20s} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
