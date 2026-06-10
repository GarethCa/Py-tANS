"""pytans — a pure-Python tabled Asymmetric Numeral Systems (tANS) coder.

tANS is the table-driven variant of Jarek Duda's Asymmetric Numeral
Systems (arXiv:1311.2540), the entropy coder behind Zstandard's Finite
State Entropy. This package provides:

- :func:`compress` / :func:`decompress` — one-shot, self-contained frames.
- :func:`compress_stream` / :func:`decompress_stream` — block-based
  streaming for large files and pipes, with bounded memory use
  (async variants live in :mod:`pytans.aio`).
- :class:`TansCoder` — a reusable coder built from symbol statistics, for
  when many messages share one table (dictionary-style usage).
- ``pytans`` CLI — compress and decompress files.
"""

from .coder import TansCoder
from .exceptions import CorruptedDataError, TansError
from .frame import compress, decompress
from .stream import DEFAULT_BLOCK_SIZE, compress_stream, decompress_stream
from .tables import normalize_counts, optimal_table_log

__version__ = "0.2.0"

__all__ = [
    "compress",
    "decompress",
    "compress_stream",
    "decompress_stream",
    "DEFAULT_BLOCK_SIZE",
    "TansCoder",
    "normalize_counts",
    "optimal_table_log",
    "TansError",
    "CorruptedDataError",
    "__version__",
]
