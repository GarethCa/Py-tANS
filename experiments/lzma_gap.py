#!/usr/bin/env python3
"""Experiments: closing the gap from the v0.3 transforms to lzma.

lzma's edge over our BWT/LZ77 pipelines comes from three places:

1. parsing  — lzma uses near-optimal price-based parsing; ours is greedy.
              Cheap step in that direction: lazy matching (1-position
              lookahead, what zlib calls it).
2. rep-offsets — lzma caches recent offsets; structured data repeats the
              same stride constantly, and a cached offset costs ~no bits.
3. context modeling — lzma codes literals conditioned on the previous
              byte. tANS tables are static, so the tANS-friendly version
              is *context splitting*: partition symbols by their context
              bucket and give each bucket its own table (two-pass,
              static, decoder-reconstructible).

Pipelines measured here (sizes include all substream frame headers):

    o0          plain pytans (baseline)
    o1          order-1 context split of raw bytes (prev byte >> 4)
    lz          v0.3 greedy LZ77 + tANS (baseline)
    lz+lazy     + lazy matching
    lz+lazy+rep + rep-offset code (bucket 0 = "same offset as last match")
    lz+all      + order-1 context-split literals
    bwt         v0.3 BWT + MTF + RLE + tANS (baseline)
    bwt+ctx     MTF symbols context-split by previous MTF symbol bucket

Usage:  python experiments/lzma_gap.py [--size BYTES]
"""

import argparse
import bz2
import json
import lzma
import random
import time
import zlib
from pathlib import Path

from pytans import compress
from pytans.transforms import _bwt, _byte_escape, _mtf, _zero_rle_split

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
    return out

# ------------------------------------------------- context splitting

def ctx_split_size(symbols, contexts, n_ctx):
    """Total compressed size of symbols partitioned by context bucket."""
    streams = [bytearray() for _ in range(n_ctx)]
    for sym, ctx in zip(symbols, contexts):
        streams[ctx].append(sym)
    return sum(len(compress(bytes(s))) for s in streams if s)


def order1_size(data):
    contexts = [0] + [b >> 4 for b in data[:-1]]
    return ctx_split_size(data, contexts, 16)

# ------------------------------------------------------------------ lz77

MIN_MATCH = 4
MAX_CHAIN = 48


def lz_parse(data, lazy=False):
    """Greedy or lazy hash-chain parse.

    Returns (literals, literal_prev_bytes, sequences); sequences are
    (lit_run, match_len, offset), final one may be literals-only.
    """
    n = len(data)
    table = {}

    def remember(pos):
        bucket = table.setdefault(data[pos:pos + MIN_MATCH], [])
        bucket.append(pos)
        if len(bucket) > 2 * MAX_CHAIN:
            del bucket[:MAX_CHAIN]

    def find(pos):
        best_len, best_off = 0, 0
        if pos + MIN_MATCH <= n:
            chain = table.get(data[pos:pos + MIN_MATCH])
            if chain:
                for j in reversed(chain[-MAX_CHAIN:]):
                    length = MIN_MATCH
                    while pos + length < n and data[j + length] == data[pos + length]:
                        length += 1
                    if length > best_len:
                        best_len, best_off = length, pos - j
                        if length >= 512:
                            break
        return best_len, best_off

    literals = bytearray()
    lit_prev = bytearray()
    sequences = []
    lit_run = 0
    i = 0
    while i < n:
        best_len, best_off = find(i)
        take_literal = best_len < MIN_MATCH
        if not take_literal and lazy and i + 1 < n:
            remember(i)
            next_len, _ = find(i + 1)
            if next_len > best_len:
                take_literal = True
        if take_literal:
            literals.append(data[i])
            lit_prev.append(data[i - 1] if i else 0)
            lit_run += 1
            if not lazy and i + MIN_MATCH <= n:
                remember(i)
            elif lazy and best_len < MIN_MATCH and i + MIN_MATCH <= n:
                remember(i)
            i += 1
            continue
        sequences.append((lit_run, best_len, best_off))
        lit_run = 0
        end = i + best_len
        step = 1 if best_len < 64 else 4
        for p in range(i if not lazy else i + 1, min(end, n - MIN_MATCH), step):
            remember(p)
        i = end
    if lit_run:
        sequences.append((lit_run, 0, 0))
    return bytes(literals), bytes(lit_prev), sequences


def lz_size(data, lazy=False, rep=False, ctx=False):
    literals, lit_prev, sequences = lz_parse(data, lazy)
    matches = [s for s in sequences if s[1]]

    buckets = bytearray()
    extra_bits = 0
    last_offset = 0
    for _, _, offset in matches:
        if rep and offset == last_offset:
            buckets.append(0)            # rep code: zero extra bits
        else:
            buckets.append(offset.bit_length())
            extra_bits += max(offset.bit_length() - 1, 0)
        last_offset = offset

    if ctx:
        lit_size = ctx_split_size(literals, [b >> 4 for b in lit_prev], 16)
    else:
        lit_size = len(compress(literals))

    return (
        lit_size
        + len(compress(_byte_escape(s[0] for s in sequences)))
        + len(compress(_byte_escape(s[1] - MIN_MATCH for s in matches)))
        + len(compress(bytes(buckets)))
        + (extra_bits + 7) // 8
    )

# ------------------------------------------------------------------- bwt

def _mtf_ctx(prev_sym):
    if prev_sym < 2:
        return prev_sym
    return min(prev_sym.bit_length(), 6) + 1   # 0,1,2..7 -> 8 buckets


def bwt_size(data, ctx=False):
    syms, runs = _zero_rle_split(_mtf(_bwt(data)[0]))
    if ctx:
        contexts = [0] + [_mtf_ctx(s) for s in syms[:-1]]
        sym_size = ctx_split_size(syms, contexts, 8)
    else:
        sym_size = len(compress(syms))
    return sym_size + len(compress(runs))

# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1 << 20)
    args = parser.parse_args()

    pipelines = [
        ("o0", lambda d: len(compress(d))),
        ("o1", order1_size),
        ("lz", lambda d: lz_size(d)),
        ("lz+lazy", lambda d: lz_size(d, lazy=True)),
        ("+rep", lambda d: lz_size(d, lazy=True, rep=True)),
        ("+ctx", lambda d: lz_size(d, lazy=True, rep=True, ctx=True)),
        ("bwt", lambda d: bwt_size(d)),
        ("bwt+ctx", lambda d: bwt_size(d, ctx=True)),
        ("zlib-9", lambda d: len(zlib.compress(d, 9))),
        ("bz2-9", lambda d: len(bz2.compress(d, 9))),
        ("lzma", lambda d: len(lzma.compress(d))),
    ]
    names = [p[0] for p in pipelines]
    print(f"{'corpus':18s} " + " ".join(f"{n:>8s}" for n in names))
    for cname, data in corpora(args.size).items():
        cells = []
        for _, fn in pipelines:
            cells.append(f"{fn(data) / len(data):>8.1%}")
        print(f"{cname:18s} " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
