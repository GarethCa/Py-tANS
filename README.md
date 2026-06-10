# Py-tANS

This repository contains a pure-Python implementation of the **tANS** (tabled
Asymmetric Numeral Systems) algorithm developed by both Yann Collet and Jarek
Duda.

Asymmetric Numeral Systems is an approach to entropy encoding discovered by
Jarek Duda, detailed in the following report:
[arXiv:1311.2540](https://arxiv.org/abs/1311.2540).

The code in this repository is an adaptation of the method of tANS used by Yann
Collet within Zstd, developed for Facebook — specifically the
[Finite State Entropy](https://github.com/Cyan4973/FiniteStateEntropy) code. The
original code detailed its usage with the C programming language, while this
repo contains a Python implementation; see Collet's
[blog series](http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html)
for a walkthrough of the method. The original exploratory Jupyter notebook lives
in [`examples/Py-tANS.ipynb`](examples/Py-tANS.ipynb).

## Algorithm Overview

tANS or FSE is a method of range coding using an Asymmetric Numeral System. A
set of states are used so that the probability of a state occurring is as close
as possible to the probability of the symbol it represents.

The action of changing states is analogous to changing state within a finite
state machine, where we output bits to a bitstream as we change state by
encoding one more symbol.

<img src=https://3.bp.blogspot.com/-2kAzQkAjifA/WXFdxA2UjYI/AAAAAAAABwo/YcQwCC7Jmm0p8x55y-d3tuBwxdk-MtnrgCPcBGAYYCw/s1600/4states.png />

The hardest part of the algorithm to understand is how symbols are encoded at
close to the number of bits we can theoretically reach — the optimal bits per
symbol given by the Shannon entropy calculation — when that entropy is a
fractional number of bits. We only ever output an integer number of bits per
symbol, so we must occasionally use fewer or more bits to encode a symbol, so
that the *average* best matches the fractional ideal.

This is achieved by outputting a variable number of bits depending on which
"subrange" of states the current state falls into. The idea is that the number
of bits output will not be optimal on a one-output basis, but instead will
average to the correct fraction.

<img src=https://3.bp.blogspot.com/-4fGLCD4S3ck/WXI2nGBruzI/AAAAAAAABww/pXffRq9_TT4IdHTKSqupSKRhGyeIEZEXgCLcBGAs/s1600/5ranges.png />

Another factor to consider is the concept of moving between states, which must
correctly tell us what symbol was last encoded. This is accomplished by
outputting the binary representation of how many states we must move as the bit
output, which requires the spacing between a symbol's states to be consistent
with the sub-range bits seen in the picture above.

To accomplish this, the states are spread across the state space in such a way
as to provide the correct gaps between sub-range values — this package uses
FSE's coprime-step spread.

<img src=http://2.bp.blogspot.com/-xbkXS6jDSCk/Uvf238sQKmI/AAAAAAAAA-Q/I0AmHbver98/s1600/16states_fastScan.png />

The spreading of symbols can be done in multiple different ways, and can even be
scrambled based on a cryptographic key, trading some compression for the
property that only a holder of the key can decode.

## Limitations

- tANS is an order-0 entropy coder: it exploits symbol frequencies, not
  repetition. Use it as the entropy stage after a match-finder (as Zstd does),
  or expect text to shrink only to its order-0 entropy.
- The encoding is fragile by design: a corrupted bit changes everything decoded
  after it. The decoder's bit-accounting and final-state checks detect
  corruption, but there is no checksum and no recovery.
- No random access, and no streaming *within* a frame: tANS decodes the
  bitstream last-in-first-out, so a frame must be decoded as a whole. Streaming
  is therefore done at block granularity — the package's streaming layer splits
  input into independently-coded block frames (the same approach Zstd takes).
- This is a readable pure-Python implementation, not a fast one.

---

# The `pytans` package

## What it can do

- **One-shot compression** — `compress()` / `decompress()` produce a
  self-contained frame that carries its own symbol table and length, so
  `decompress(compress(x)) == x` for *any* bytes, with no other bookkeeping.
- **Raw-storage fallback** — input that would not shrink (random or
  already-compressed data, tiny inputs) is stored verbatim inside the frame, so
  output is never more than a few bytes larger than the input.
- **Reusable coders with shared tables** — build a `TansCoder` once from sample
  data or explicit counts and code any number of messages with it. Table
  construction is canonical (deterministic), so an encoder and decoder built
  independently from the same statistics interoperate — useful when many short
  messages share one distribution and you don't want a table in every payload.
- **Streaming for big files and pipes** — `compress_stream()` /
  `decompress_stream()` process data in independent blocks (128 KiB by
  default), so memory stays bounded no matter the file size, each block gets a
  table tuned to its own statistics, and the receiver can decode block by block
  as data arrives. Asyncio variants in `pytans.aio` work with
  `asyncio.StreamReader`/`StreamWriter` and `aiofiles` handles.
- **Near-entropy compression** — within ~2 % of the order-0 Shannon bound on
  skewed data (enforced by the test suite).
- **Tunable precision/size trade-off** — `table_log` (4–15) sets the state-table
  size to `2**table_log`; an FSE-style heuristic picks a sensible value
  automatically from the input size and alphabet.
- **Frequency tooling** — `normalize_counts()` scales raw counts to a
  power-of-two total while guaranteeing rare symbols stay encodable;
  `optimal_table_log()` exposes the auto-sizing heuristic.
- **Corruption detection** — truncated or malformed frames, bad headers and
  bitstreams that fail the decoder's final-state / bit-accounting invariants all
  raise `CorruptedDataError` rather than returning wrong data silently.
- **A command-line tool** — `pytans compress` / `pytans decompress` for files or
  stdin/stdout pipelines.
- **Clean packaging** — typed (`py.typed`), zero runtime dependencies,
  Python 3.9+.

What it deliberately does *not* do, because tANS itself doesn't: see
[Limitations](#limitations).

## Installation

```sh
pip install .            # from a checkout
pip install -e '.[dev]'  # development install with pytest
```

## Quick start

```python
>>> import pytans
>>> text = b"how much wood would a woodchuck chuck if a woodchuck could chuck wood " * 100
>>> blob = pytans.compress(text)
>>> len(text), len(blob)
(7000, 2948)                     # 42% of the original (order-0 entropy of this text)
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

## Usage

### One-shot frames

`compress` returns a frame containing a magic number, the normalised symbol
table, the original length and the tANS payload. `decompress` needs nothing
else:

```python
blob = pytans.compress(data)                # auto-sized table, capped at 2**12 states
blob = pytans.compress(data, table_log=8)   # force a 256-state table (smaller header)
data = pytans.decompress(blob)
```

### Reusable coder: share one table across many messages

A frame's symbol table costs up to 3 bytes per distinct symbol — significant for
short messages. If many messages share a distribution, build the table once,
transmit/agree on the statistics out of band, and send bare payloads:

```python
from pytans import TansCoder

# Build from sample data (or pass explicit counts: TansCoder({101: 60, 116: 40}))
encoder = TansCoder.from_data(training_corpus)

payload, bit_length = encoder.encode(b"chuck wood much")
# -> 8 bytes, 60 bits: no per-message table overhead

# Elsewhere: same stats in, identical tables out (construction is canonical)
decoder = TansCoder(encoder.normalized_counts, encoder.table_log)
message = decoder.decode(payload, bit_length, length=15)
```

`encode` returns `(payload, bit_length)`; decoding needs the payload, the exact
bit length (the final byte is zero-padded) and the message length in symbols, so
store or transmit those two integers alongside the payload — that is exactly
what the frame format does for you.

A coder can encode anything drawn from its alphabet — encoding a byte it has
never seen raises `ValueError`:

```python
encoder.symbols             # byte values the coder knows, ascending
encoder.normalized_counts   # {byte: slots} summing to encoder.table_size
encoder.table_log           # chosen automatically here (11 for the corpus above)
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

Any binary file-like objects work (files, pipes, sockets, `io.BytesIO`), short
reads are handled, and `decompress_stream` also accepts a plain one-shot
`compress()` frame. `block_size` trades header overhead (smaller blocks pay a
per-block symbol table) against memory and decode latency.

For async code, `pytans.aio` mirrors the same two functions and duck-types its
sources and sinks — `asyncio.StreamReader`/`StreamWriter` pairs, `aiofiles`
handles, and even plain sync file objects all work, and `drain()` is awaited
for backpressure when present:

```python
from pytans import aio

async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    await aio.decompress_stream(reader, writer)
```

The coding itself is CPU-bound and runs inline; wrap calls in
`loop.run_in_executor` if the event loop must stay free during large blocks.

### Controlling the tables

```python
from pytans import normalize_counts, optimal_table_log

# Scale raw counts to sum to 2**table_log; rare symbols never drop to zero.
normalize_counts({65: 900, 66: 90, 67: 9, 68: 1}, table_log=5)
# -> {65: 28, 66: 2, 67: 1, 68: 1}

# The auto-sizing heuristic: bigger inputs and alphabets earn bigger tables.
optimal_table_log(sample_size=100_000, alphabet_size=4)   # -> 12
```

Larger `table_log` = closer fit to the true distribution (better ratio) but a
bigger table to build and, for frames, a bigger header. The default cap of 12
(4096 states) is plenty for byte data; the full range is 4–15.

### Handling errors

All package errors derive from `TansError` (a `ValueError`); anything that
indicates damaged input data is the subclass `CorruptedDataError`:

```python
from pytans import CorruptedDataError, TansError

try:
    data = pytans.decompress(blob)
except CorruptedDataError as err:
    ...   # truncated, tampered with, or not a pytans frame
```

### Command line

```sh
pytans compress war-and-peace.txt              # writes war-and-peace.txt.tans
pytans decompress war-and-peace.txt.tans      # restores war-and-peace.txt
pytans compress big.csv -o out.tans --table-log 12 --block-size 262144
pytans compress - < input > output.tans       # stdin/stdout pipelines
pytans decompress output.tans -o - | head
```

Existing outputs are never overwritten without `-f/--force`.

## API reference

| Name | Description |
| --- | --- |
| `compress(data, table_log=None, max_table_log=12) -> bytes` | Compress bytes into a self-contained frame; stores raw if compression wouldn't help. |
| `decompress(blob) -> bytes` | Restore the original bytes from a frame. |
| `compress_stream(src, dst, *, block_size=131072, table_log=None, max_table_log=12) -> (in, out)` | Stream-compress a file-like object block by block. |
| `decompress_stream(src, dst) -> (in, out)` | Stream-decompress; also accepts a single one-shot frame. |
| `pytans.aio.compress_stream` / `decompress_stream` | Async equivalents for asyncio/aiofiles sources and sinks. |
| `DEFAULT_BLOCK_SIZE` | Default streaming block size (128 KiB). |
| `TansCoder(counts, table_log=None)` | Build a coder from raw or normalised per-byte counts. |
| `TansCoder.from_data(sample, table_log=None)` | Build a coder from the byte statistics of a sample. |
| `TansCoder.encode(data) -> (payload, bit_length)` | Encode bytes drawn from the coder's alphabet. |
| `TansCoder.decode(payload, bit_length, length) -> bytes` | Decode `length` symbols; validates stream integrity. |
| `TansCoder.table_log` / `.table_size` / `.symbols` / `.normalized_counts` | Inspect the coder's table. |
| `normalize_counts(counts, table_log) -> dict` | Scale counts to sum to `2**table_log`, keeping every symbol ≥ 1. |
| `optimal_table_log(sample_size, alphabet_size, max_table_log=12) -> int` | FSE-style automatic table sizing. |
| `TansError` | Base error (subclass of `ValueError`). |
| `CorruptedDataError` | Frame/bitstream failed validation. |

The exact frame layout is documented in
[`src/pytans/frame.py`](src/pytans/frame.py).

## Development

```sh
pip install -e '.[dev]'
pytest
```
