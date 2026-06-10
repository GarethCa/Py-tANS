# Py-tANS

A pure-Python implementation of **tANS** (tabled Asymmetric Numeral Systems), the
entropy coder developed by Jarek Duda and Yann Collet.

Asymmetric Numeral Systems is an approach to entropy encoding discovered by Jarek
Duda, detailed in [arXiv:1311.2540](https://arxiv.org/abs/1311.2540). The code in
this repository follows the tANS construction used by Yann Collet's
[Finite State Entropy](https://github.com/Cyan4973/FiniteStateEntropy) (the entropy
stage of Zstandard) — see his
[blog series](http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html)
for a walkthrough. The original exploratory Jupyter notebook lives in
[`examples/Py-tANS.ipynb`](examples/Py-tANS.ipynb).

## Installation

```sh
pip install .          # from a checkout
pip install -e '.[dev]'  # development install with pytest
```

Requires Python 3.9+. No runtime dependencies.

## Usage

### One-shot compression

`compress` produces a self-contained frame: the symbol table, original length and
payload travel together, and incompressible input is stored raw so output is never
much larger than the input.

```python
import pytans

blob = pytans.compress(b"how much wood would a woodchuck chuck" * 100)
data = pytans.decompress(blob)
```

### Reusable coder (shared tables)

When many short messages share one symbol distribution, build the table once and
ship only the payloads — the decoder rebuilds an identical table from the same
statistics, since table construction is canonical.

```python
from pytans import TansCoder

encoder = TansCoder.from_data(training_sample)            # or TansCoder(counts)
payload, bit_length = encoder.encode(message)

decoder = TansCoder(encoder.normalized_counts, encoder.table_log)
message = decoder.decode(payload, bit_length, len(message))
```

`table_log` controls the state-table size (`2**table_log` states, 4–15): larger
tables track the symbol distribution more precisely at the cost of a larger
table/header.

### Command line

```sh
pytans compress war-and-peace.txt            # writes war-and-peace.txt.tans
pytans decompress war-and-peace.txt.tans
pytans compress - < input > output.tans      # stdin/stdout
```

### Errors

Malformed frames, truncated payloads and failed integrity checks raise
`pytans.CorruptedDataError`; all package errors derive from `pytans.TansError`
(a `ValueError`).

## Algorithm Overview

tANS or FSE is a method of range coding using an Asymmetric Numeral System. A set
of states are used so that the probability of a state occurring is as close as
possible to the probability of the symbol it represents.

The action of changing states is analogous to changing state within a finite state
machine, where we output bits to a bitstream as we change state by encoding one
more symbol.

<img src=https://3.bp.blogspot.com/-2kAzQkAjifA/WXFdxA2UjYI/AAAAAAAABwo/YcQwCC7Jmm0p8x55y-d3tuBwxdk-MtnrgCPcBGAYYCw/s1600/4states.png />

The hardest part of the algorithm is encoding symbols at close to their Shannon
entropy when that entropy is a fractional number of bits. We only ever output an
integer number of bits per symbol, so we must occasionally use fewer or more bits
for a symbol so that the *average* matches the fractional ideal.

This is achieved by outputting a variable number of bits depending on which
"subrange" of states the current state falls into: no single output is optimal,
but the outputs average to the correct fraction.

<img src=https://3.bp.blogspot.com/-4fGLCD4S3ck/WXI2nGBruzI/AAAAAAAABww/pXffRq9_TT4IdHTKSqupSKRhGyeIEZEXgCLcBGAs/s1600/5ranges.png />

Decoding relies on moving between states in a way that identifies which symbol was
last encoded. To make that work, each symbol's states are spread across the state
space with consistent gaps matching its sub-range bit counts — this package uses
FSE's coprime-step spread.

<img src=http://2.bp.blogspot.com/-xbkXS6jDSCk/Uvf238sQKmI/AAAAAAAAA-Q/I0AmHbver98/s1600/16states_fastScan.png />

## Limitations

- tANS is an order-0 entropy coder: it exploits symbol frequencies, not
  repetition. Use it as the entropy stage after a match-finder (as Zstd does), or
  expect text to shrink only to its order-0 entropy.
- The encoding is fragile by design: a corrupted bit changes everything decoded
  after it. The decoder's bit-accounting and final-state checks detect corruption,
  but there is no checksum and no recovery.
- No random access or streaming: tANS decodes the bitstream last-in-first-out, so
  a frame must be decoded as a whole.
- This is a readable pure-Python implementation, not a fast one.

## Development

```sh
pip install -e '.[dev]'
pytest
```
