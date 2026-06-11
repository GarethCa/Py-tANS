#!/usr/bin/env python3
"""Regenerate the README's "How it compares" table.

Benchmarks pytans against Python's stdlib compressors (zlib, bz2, lzma)
across several kinds of data, reporting compressed size as a percentage
of the original alongside the order-0 Shannon entropy floor — the
theoretical best for any pure entropy coder.

Corpora that cannot be built on this machine (no network for the
Gutenberg text, no macOS system sounds, ...) are skipped with a note, so
the script runs anywhere but reproduces the full table only where the
README's numbers were measured.

Usage:
    python examples/benchmark.py
"""

import bz2
import glob
import json
import lzma
import math
import random
import sysconfig
import urllib.error
import urllib.request
import zlib
from collections import Counter
from io import BytesIO
from pathlib import Path

from pytans import compress, compress_stream

GUTENBERG_TEXT = "https://www.gutenberg.org/files/2600/2600-0.txt"  # War and Peace
PY_SOURCE_BUDGET = 3_000_000


def english_text():
    cache = Path("/tmp/pytans-bench-wnp.txt")
    if not cache.exists():
        with urllib.request.urlopen(GUTENBERG_TEXT, timeout=30) as response:
            cache.write_bytes(response.read())
    return cache.read_bytes()


def python_source():
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    corpus = bytearray()
    for path in sorted(stdlib.rglob("*.py")):
        try:
            corpus += path.read_bytes()
        except OSError:
            continue
        if len(corpus) >= PY_SOURCE_BUDGET:
            break
    return bytes(corpus[:PY_SOURCE_BUDGET])


def sorted_dictionary():
    return Path("/usr/share/dict/words").read_bytes()


def pcm_audio():
    import aifc  # deprecated; removed in Python 3.13

    pcm = bytearray()
    for path in sorted(glob.glob("/System/Library/Sounds/*.aiff")):
        with aifc.open(path) as audio:
            pcm += audio.readframes(audio.getnframes())
    if not pcm:
        raise FileNotFoundError("no system sounds found")
    return bytes(pcm)


def json_logs(rng):
    lines = []
    for i in range(20_000):
        lines.append(json.dumps({
            "ts": 1_750_000_000 + i,
            "level": rng.choice(["INFO"] * 8 + ["WARN", "ERROR"]),
            "service": rng.choice(["auth", "billing", "api"]),
            "latency_ms": rng.randrange(1, 500),
            "ok": rng.random() < 0.95,
            "request_id": f"{rng.randrange(16**12):012x}",
        }))
    return "\n".join(lines).encode()


def build_corpora():
    rng = random.Random(0)
    builders = [
        ("English text (War and Peace)", english_text),
        ("Python source code", python_source),
        ("Sorted dictionary", sorted_dictionary),
        ("PCM audio (system sounds)", pcm_audio),
        ("JSON logs (synthetic)", lambda: json_logs(rng)),
        ("Skewed bytes, no repetition",
         lambda: bytes(rng.choices(range(8), weights=[40, 20, 10, 8, 6, 5, 4, 3],
                                   k=4_000_000))),
        ("Random noise", lambda: rng.randbytes(2_000_000)),
    ]
    for name, build in builders:
        try:
            yield name, build()
        except (OSError, ImportError, urllib.error.URLError) as exc:
            print(f"  (skipping {name}: {exc})")


def entropy_floor(data):
    counts = Counter(data)
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in counts.values()) / 8


def pytans_ratio(data):
    out = BytesIO()
    compress_stream(BytesIO(data), out)
    return out.tell() / len(data)


def pytans_bwt_ratio(data):
    # Whole-buffer frame: gives the BWT full context, like bzip2's big blocks.
    return len(compress(data, transform="bwt")) / len(data)


def main():
    rows = []
    for name, data in build_corpora():
        rows.append((
            name,
            len(data),
            entropy_floor(data),
            pytans_ratio(data),
            pytans_bwt_ratio(data),
            len(zlib.compress(data, 9)) / len(data),
            len(bz2.compress(data, 9)) / len(data),
            len(lzma.compress(data)) / len(data),
        ))

    print()
    print("| Data | Size | Floor | pytans | pytans bwt | zlib -9 | bz2 -9 | lzma |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, n, floor, pt, ptb, z, b, l in rows:
        cells = " | ".join(f"{v:.1%}".replace("%", " %") for v in (floor, pt, ptb, z, b, l))
        print(f"| {name} | {n / 1e6:.1f} MB | {cells} |")


if __name__ == "__main__":
    main()
