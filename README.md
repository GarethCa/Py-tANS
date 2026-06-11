<h1 align="center">Py-tANS</h1>

<p align="center"><em>A pure-Python implementation of tabled Asymmetric Numeral Systems —
the entropy coder behind Zstandard.</em></p>

<p align="center">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Version"     src="https://img.shields.io/badge/version-0.2.0-informational">
  <img alt="Dependencies" src="https://img.shields.io/badge/dependencies-none-success">
  <img alt="Typed"       src="https://img.shields.io/badge/typing-py.typed-success">
</p>

This repository contains an implementation of the **tANS** algorithm developed by
both Yann Collet and Jarek Duda.

Asymmetric Numeral Systems is an approach to entropy encoding discovered by
**Jarek Duda**, detailed in his report
**[arXiv:1311.2540 — *Asymmetric numeral systems*](https://arxiv.org/abs/1311.2540)**.
The code here is an adaptation of the method of tANS used by **Yann Collet**
within Zstd — specifically the
[Finite State Entropy](https://github.com/Cyan4973/FiniteStateEntropy) code. The
original details its usage in C, while this repo contains a Python implementation;
Collet's
[blog series](http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html)
walks through the method. The original exploratory notebook is preserved in
[`examples/Py-tANS.ipynb`](examples/Py-tANS.ipynb).

---

**Contents** · [How the algorithm works](#-how-the-algorithm-works) ·
[Limitations](#-limitations) · [Features](#-features) · [Installation](#-installation) ·
[Quick start](#-quick-start) · [How it compares](#-how-it-compares) ·
[Usage guide](#-usage-guide) · [Command line](#-command-line) ·
[API reference](#-api-reference) · [Development](#-development)

---

## 🧮 How the algorithm works

tANS (or FSE) is a method of range coding using an Asymmetric Numeral System. A
set of states is used so that the probability of a state occurring is as close as
possible to the probability of the symbol it represents.

The action of changing states is analogous to changing state within a finite
state machine, where we output bits to a bitstream as we change state by encoding
one more symbol.

<p align="center"><img src="https://3.bp.blogspot.com/-2kAzQkAjifA/WXFdxA2UjYI/AAAAAAAABwo/YcQwCC7Jmm0p8x55y-d3tuBwxdk-MtnrgCPcBGAYYCw/s1600/4states.png"></p>

The hardest part of the algorithm to understand is how symbols are encoded at
close to the theoretical optimum — the bits per symbol given by the Shannon
entropy — when that optimum is a *fractional* number of bits. We can only ever
output an integer number of bits per symbol, so we must occasionally use fewer or
more bits to encode a symbol, such that the **average** matches the fractional
ideal.

This is achieved by outputting a variable number of bits depending on which
"subrange" of states the current state falls into. No single output is optimal on
its own, but the outputs average to the correct fraction:

<p align="center"><img src="https://3.bp.blogspot.com/-4fGLCD4S3ck/WXI2nGBruzI/AAAAAAAABww/pXffRq9_TT4IdHTKSqupSKRhGyeIEZEXgCLcBGAs/s1600/5ranges.png"></p>

Moving between states must also correctly tell the decoder which symbol was last
encoded. This is accomplished by outputting the binary representation of how many
states we must move, which requires the spacing between a symbol's states to be
consistent with its sub-range bit counts. To accomplish this, states are *spread*
across the state space so as to provide the correct gaps — this package uses
FSE's coprime-step spread:

<p align="center"><img src="http://2.bp.blogspot.com/-xbkXS6jDSCk/Uvf238sQKmI/AAAAAAAAA-Q/I0AmHbver98/s1600/16states_fastScan.png"></p>

The spreading can be done in different ways, and can even be scrambled based on a
cryptographic key — trading some compression for the property that only a holder
of the key can decode.

## ⚠️ Limitations

These are properties of tANS itself, not just of this implementation:

| | |
| --- | --- |
| **Order-0 only** | tANS exploits symbol *frequencies*, not repetition. The package offers opt-in BWT/LZ77 modeling transforms for repetitive data; without them, expect text to shrink only to its order-0 entropy. |
| **Fragile streams** | One corrupted bit changes everything decoded after it. The decoder *detects* corruption (final-state + bit-accounting checks) but cannot correct it — layer a checksum/ECC on top if you need that. |
| **LIFO decoding** | A frame decodes last-in-first-out and only as a whole; no random access. Streaming is therefore done at *block* granularity, like Zstd. |
| **Pure Python** | Written to be readable, not fast. |

---

# 📦 The `pytans` package

## ✨ Features

| | Feature | What you get |
| --- | --- | --- |
| 🗜️ | **One-shot compression** | `compress()` / `decompress()` — self-contained frames carrying their own symbol table and length: `decompress(compress(x)) == x` for *any* bytes, no bookkeeping. |
| 🌊 | **Streaming** | `compress_stream()` / `decompress_stream()` — independent 128 KiB blocks, so multi-GB files compress in constant memory and each block's table adapts to its local statistics. |
| ⚡ | **Async** | `pytans.aio` mirrors the streaming API for `asyncio.StreamReader`/`StreamWriter`, `aiofiles` handles, and plain files alike — `drain()` is awaited for backpressure. |
| ♻️ | **Reusable coders** | `TansCoder` is built once from sample data or counts; construction is canonical, so an encoder and decoder built independently from the same statistics interoperate — no table in every payload. |
| 🔁 | **Repetition transforms (opt-in)** | `transform="bwt"` (Burrows-Wheeler + MTF + RLE, bzip2's pipeline) or `transform="lz77"` (match-finding, zlib/zstd's shape) turn repetition into symbol skew tANS can use — text drops from 58 % to 27 %. Kept per block only when actually smaller. |
| 🪨 | **Raw fallback** | Incompressible input is stored verbatim inside the frame: output never grows by more than a few header bytes. |
| 🎯 | **Near-entropy ratios** | Within ~2 % of the order-0 Shannon bound on skewed data (enforced by the test suite); ≲0.1 % at default table sizes. |
| 🎛️ | **Tunable tables** | `table_log` 4–15 sets the precision/size trade-off; an FSE-style heuristic auto-sizes it from input size and alphabet. |
| 🛡️ | **Corruption detection** | Malformed, truncated or tampered input raises `CorruptedDataError` — never silent garbage. |
| ⌨️ | **CLI** | `pytans compress` / `pytans decompress` for files and shell pipelines. |

## 🚀 Installation

```sh
pip install .            # from a checkout
pip install -e '.[dev]'  # development install with pytest
```

Python 3.9+, no runtime dependencies.

## ⏱️ Quick start

```python
>>> import pytans
>>> text = b"how much wood would a woodchuck chuck if a woodchuck could chuck wood " * 100
>>> blob = pytans.compress(text)
>>> len(text), len(blob)
(7000, 2948)                     # 42% — the order-0 entropy of this text
>>> pytans.decompress(blob) == text
True
```

Incompressible input falls back to raw storage instead of growing:

```python
>>> import random
>>> noise = random.Random(0).randbytes(10_000)
>>> len(pytans.compress(noise))
10008                            # 8 bytes of header, nothing lost trying
```

## 📊 How it compares

Compressed size as a percentage of the original (smaller is better), measured
against Python's stdlib compressors across different kinds of data. *Floor* is
the order-0 Shannon entropy of each input — the theoretical best any pure
entropy coder can do:

| Data | Size | Floor | **pytans** | **pytans `bwt`** | zlib&nbsp;-9 | bz2&nbsp;-9 | lzma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English text (*War and Peace*) | 3.4 MB | 57.9 % | 58.1 % | 26.8 % | 36.4 % | **26.4 %** | 27.8 % |
| Python source code | 3.0 MB | 56.6 % | 56.4 % | 20.1 % | 22.9 % | 18.9 % | **18.3 %** |
| Sorted dictionary (`/usr/share/dict/words`) | 2.5 MB | 54.1 % | 52.7 % | 37.2 % | 30.2 % | 34.4 % | **25.6 %** |
| PCM audio (16-bit system sounds) | 4.8 MB | 72.1 % | 70.3 % | 64.0 % | 64.5 % | 62.1 % | **48.3 %** |
| JSON logs (synthetic) | 2.3 MB | 60.5 % | 60.6 % | 10.5 % | 12.6 % | **9.7 %** | 10.4 % |
| Skewed bytes, *no repetition* | 4.0 MB | 30.7 % | **30.7 %** | **30.7 %** | 36.8 % | 36.1 % | 34.0 % |
| Random noise | 2.0 MB | 100 % | 100.0 % | 100.0 % | 100.0 % | 100.4 % | 100.0 % |

How to read this — it is really comparing two different jobs:

- **pytans tracks the entropy floor everywhere** (sometimes slightly beating
  the *global* floor, because per-block tables adapt to local statistics).
  That is the whole job of an entropy coder, done at full efficiency.
- **zlib/bz2/lzma are complete pipelines** — a *model* (LZ77 match-finding,
  Burrows–Wheeler context sorting) feeding an entropy coder. On text, code,
  logs and audio, almost all of their win comes from the modeling stage.
- **The opt-in transforms add that modeling stage**: `transform="bwt"`
  (bzip2's recipe with tANS as the entropy coder) beats zlib everywhere
  above and effectively ties bz2 on text; `transform="lz77"` is the
  zstd shape and wins where LZ beats BWT (the sorted dictionary: 31.9 %).
- **When there is nothing to model, tANS wins.** The "skewed bytes" row has
  frequency bias but zero repetition, reducing everyone to pure entropy
  coding: pytans lands exactly on the floor while zlib pays Huffman's
  integer-bit penalty (36.8 % vs 30.7 %) — precisely the gap described in
  Duda's paper, and why Zstandard pairs its match-finder with FSE rather
  than Huffman.
- **On incompressible data everyone converges to ~100 %**; pytans's raw
  fallback caps the overhead at a few header bytes.

Speed is deliberately omitted: those are decades-tuned C libraries and this is
readable pure Python (~3 MiB/s). In C, FSE famously *outruns* zlib's entropy
stage — speed is the reason it exists.

Regenerate this table with
[`examples/benchmark.py`](examples/benchmark.py) (corpora that can't be built
on your machine are skipped).

## 📖 Usage guide

### One-shot frames

`compress` returns a frame containing a magic number, the normalised symbol
table, the original length and the tANS payload — `decompress` needs nothing
else:

```python
blob = pytans.compress(data)                # auto-sized table, capped at 2**12 states
blob = pytans.compress(data, table_log=8)   # force a 256-state table (smaller header)
data = pytans.decompress(blob)
```

### Streaming big files

The streaming layer chops input into blocks and writes one self-contained frame
per block, so a multi-gigabyte file compresses in constant memory:

```python
from pytans import compress_stream, decompress_stream

with open("big.log", "rb") as src, open("big.log.tans", "wb") as dst:
    bytes_in, bytes_out = compress_stream(src, dst)        # block_size=128 KiB

with open("big.log.tans", "rb") as src, open("big.log", "wb") as dst:
    decompress_stream(src, dst)
```

Any binary file-like objects work (files, pipes, sockets, `io.BytesIO`); short
reads are handled, and `decompress_stream` also accepts a plain one-shot
`compress()` frame. `block_size` trades per-block header overhead against memory
use and decode latency.

For async code, `pytans.aio` mirrors the same two functions and duck-types its
sources and sinks:

```python
from pytans import aio

async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    await aio.decompress_stream(reader, writer)
```

The coding itself is CPU-bound and runs inline; wrap calls in
`loop.run_in_executor` if the event loop must stay free during large blocks.

### Repetitive data: the BWT and LZ77 transforms

Plain tANS cannot see repetition. The opt-in transforms reshape the input so
repetition *becomes* symbol skew the entropy stage can exploit:

```python
blob = pytans.compress(text, transform="bwt")    # bzip2's pipeline, tANS-coded
blob = pytans.compress(text, transform="lz77")   # zlib/zstd's pipeline shape
pytans.decompress(blob)                          # transform recorded in the frame
```

- **`"bwt"`** — Burrows-Wheeler transform + move-to-front + zero-run
  splitting. The strongest choice for text, source code and logs.
- **`"lz77"`** — greedy match-finding emitting zstd-style substreams
  (literals / run lengths / match lengths / offset buckets), each compressed
  with its own fitted tANS table.

A transformed frame is kept **only when it is actually smaller** than the
plain one, so enabling a transform never costs more than CPU — on
non-repetitive data the output is byte-identical to plain mode. Streaming
applies the transform per block (`compress_stream(..., transform="bwt")`);
larger blocks give BWT more context, so consider `--block-size` 512 KiB–1 MiB
for bz2-like ratios. The BWT suffix sort automatically uses numpy when
installed (much faster); otherwise a pure-Python sort is used.

### Reusable coder: one table, many messages

A frame's symbol table costs up to 3 bytes per distinct symbol — significant for
short messages. If many messages share a distribution, build the table once,
agree on the statistics out of band, and send bare payloads:

```python
from pytans import TansCoder

# Build from sample data (or explicit counts: TansCoder({101: 60, 116: 40}))
encoder = TansCoder.from_data(training_corpus)

payload, bit_length = encoder.encode(b"chuck wood much")
# -> 8 bytes, 60 bits: no per-message table overhead

# Elsewhere: same stats in, identical tables out (construction is canonical)
decoder = TansCoder(encoder.normalized_counts, encoder.table_log)
message = decoder.decode(payload, bit_length, length=15)
```

`encode` returns `(payload, bit_length)`; decoding needs the payload, the exact
bit length (the final byte is zero-padded) and the message length in symbols —
store those two integers alongside the payload, which is exactly what the frame
format does for you.

A coder encodes anything drawn from its alphabet; a byte it has never seen
raises `ValueError`. Inspect it via:

```python
encoder.symbols             # byte values the coder knows, ascending
encoder.normalized_counts   # {byte: slots} summing to encoder.table_size
encoder.table_log           # chosen automatically here
```

### Controlling the tables

```python
from pytans import normalize_counts, optimal_table_log

# Scale raw counts to sum to 2**table_log; rare symbols never drop to zero.
normalize_counts({65: 900, 66: 90, 67: 9, 68: 1}, table_log=5)
# -> {65: 28, 66: 2, 67: 1, 68: 1}

# The auto-sizing heuristic: bigger inputs and alphabets earn bigger tables.
optimal_table_log(sample_size=100_000, alphabet_size=4)   # -> 12
```

Larger `table_log` = a closer fit to the true distribution, but the returns
diminish fast — measured on skewed 8-symbol data:

| table_log | states | overhead vs Shannon entropy |
| ---: | ---: | ---: |
| 4 | 16 | +2.8 % |
| 6 | 64 | +1.7 % |
| 8 | 256 | +0.10 % |
| 12 | 4096 | +0.003 % |

The default cap of 12 is plenty for byte data; small inputs automatically get
smaller tables so the header doesn't outweigh the gain.

### Handling errors

All package errors derive from `TansError` (a `ValueError`); anything that
indicates damaged input is the subclass `CorruptedDataError`:

```python
from pytans import CorruptedDataError

try:
    data = pytans.decompress(blob)
except CorruptedDataError:
    ...   # truncated, tampered with, or not a pytans frame
```

## ⌨️ Command line

```sh
pytans compress war-and-peace.txt             # writes war-and-peace.txt.tans
pytans decompress war-and-peace.txt.tans      # restores war-and-peace.txt
pytans compress - < input > output.tans       # stdin/stdout pipelines
pytans decompress output.tans -o - | head
```

| Option | Applies to | Meaning |
| --- | --- | --- |
| `-o`, `--output` | both | Output path (`-` for stdout); defaults to adding/stripping `.tans` |
| `-f`, `--force` | both | Overwrite an existing output file |
| `--table-log N` | compress | Force a `2**N`-state table (4–15; default: auto) |
| `--block-size BYTES` | compress | Uncompressed bytes per streaming block (default 131072) |
| `--transform {bwt,lz77}` | compress | Modeling stage for repetitive data; kept per block only when it shrinks the output |

Files are processed through the streaming layer, so memory stays bounded
regardless of file size, and partial output is removed if the input turns out to
be corrupt.

## 🔍 API reference

| Name | Description |
| --- | --- |
| `compress(data, table_log=None, max_table_log=12, transform=None) -> bytes` | Compress bytes into a self-contained frame; stores raw if compression wouldn't help. |
| `decompress(blob) -> bytes` | Restore the original bytes from a frame (any mode, transforms included). |
| `compress_stream(src, dst, *, block_size=131072, table_log=None, max_table_log=12, transform=None) -> (in, out)` | Stream-compress a binary file-like, block by block. |
| `decompress_stream(src, dst) -> (in, out)` | Stream-decompress; also accepts a single one-shot frame. |
| `pytans.aio.compress_stream` / `decompress_stream` | Async equivalents for asyncio / aiofiles sources and sinks. |
| `TansCoder(counts, table_log=None)` | Build a coder from raw or normalised per-byte counts. |
| `TansCoder.from_data(sample, table_log=None)` | Build a coder from the byte statistics of a sample. |
| `TansCoder.encode(data) -> (payload, bit_length)` | Encode bytes drawn from the coder's alphabet. |
| `TansCoder.decode(payload, bit_length, length) -> bytes` | Decode `length` symbols; validates stream integrity. |
| `TansCoder.table_log` / `.table_size` / `.symbols` / `.normalized_counts` | Inspect the coder's table. |
| `normalize_counts(counts, table_log) -> dict` | Scale counts to sum to `2**table_log`, keeping every symbol ≥ 1. |
| `optimal_table_log(sample_size, alphabet_size, max_table_log=12) -> int` | FSE-style automatic table sizing. |
| `DEFAULT_BLOCK_SIZE` | Default streaming block size (128 KiB). |
| `TRANSFORMS` | Available transform names (`"bwt"`, `"lz77"`); details in [`src/pytans/transforms.py`](src/pytans/transforms.py). |
| `TansError` / `CorruptedDataError` | Base error (a `ValueError`) / damaged-input error. |

Exact byte layouts are documented in
[`src/pytans/frame.py`](src/pytans/frame.py) (one-shot frames) and
[`src/pytans/stream.py`](src/pytans/stream.py) (stream container).

## 🛠️ Development

```sh
pip install -e '.[dev]'
pytest        # 100 tests
```
