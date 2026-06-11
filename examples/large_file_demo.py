#!/usr/bin/env python3
"""Try pytans on your own large files.

Streams each file through the block-based compressor, verifies the
roundtrip is byte-identical, and reports ratio, throughput and how close
the result sits to the file's order-0 entropy (the theoretical floor for
tANS). zlib is shown alongside for context — it adds match-finding on
top of entropy coding, so beating it on text is not expected.

Usage:
    python examples/large_file_demo.py FILE [FILE ...]
    python examples/large_file_demo.py FILE --keep        # keep FILE.tans
    python examples/large_file_demo.py FILE --block-size 262144 --table-log 12
"""

import argparse
import math
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path

from pytans import compress_stream, decompress_stream


def order0_entropy_bits(path: Path) -> float:
    counts: Counter = Counter()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            counts.update(chunk)
            total += len(chunk)
    if total == 0:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def zlib_size(path: Path) -> int:
    compressor = zlib.compressobj(level=6)
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            size += len(compressor.compress(chunk))
    return size + len(compressor.flush())


def demo(path: Path, block_size: int, table_log, keep: bool) -> bool:
    original_size = path.stat().st_size
    packed_path = path.with_name(path.name + ".tans")
    print(f"\n=== {path} ({original_size:,} bytes) ===")

    start = time.perf_counter()
    with open(path, "rb") as src, open(packed_path, "wb") as dst:
        _, packed_size = compress_stream(
            src, dst, block_size=block_size, table_log=table_log
        )
    compress_secs = time.perf_counter() - start

    start = time.perf_counter()
    with open(packed_path, "rb") as src, tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False
    ) as dst:
        restored_path = Path(dst.name)
        decompress_stream(src, dst)
    decompress_secs = time.perf_counter() - start

    identical = restored_path.read_bytes() == path.read_bytes()
    restored_path.unlink()
    if not keep:
        packed_path.unlink()

    entropy = order0_entropy_bits(path)
    floor = int(entropy * original_size / 8)
    z = zlib_size(path)
    mib = original_size / (1 << 20)

    print(f"  pytans:   {packed_size:>12,} bytes  ({packed_size / original_size:6.1%})")
    print(f"  entropy:  {floor:>12,} bytes  ({floor / original_size:6.1%})"
          f"  <- order-0 floor at {entropy:.3f} bits/byte")
    print(f"  zlib -6:  {z:>12,} bytes  ({z / original_size:6.1%})"
          f"  <- has match-finding, different game")
    print(f"  speed:    {mib / compress_secs:6.2f} MiB/s compress, "
          f"{mib / decompress_secs:6.2f} MiB/s decompress")
    print(f"  roundtrip: {'byte-identical ✓' if identical else 'MISMATCH ✗'}"
          + (f"   (kept {packed_path})" if keep else ""))
    return identical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--block-size", type=int, default=128 * 1024)
    parser.add_argument("--table-log", type=int, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the .tans output")
    args = parser.parse_args()

    ok = True
    for path in args.files:
        if not path.is_file():
            print(f"skipping {path}: not a file", file=sys.stderr)
            continue
        ok &= demo(path, args.block_size, args.table_log, args.keep)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
