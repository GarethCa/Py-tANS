"""Command-line interface: ``pytans compress`` / ``pytans decompress``.

Files are processed through the block-based streaming layer, so memory
use stays bounded by the block size regardless of file size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .exceptions import TansError
from .stream import DEFAULT_BLOCK_SIZE, compress_stream, decompress_stream
from .tables import DEFAULT_MAX_TABLE_LOG, MAX_TABLE_LOG, MIN_TABLE_LOG

SUFFIX = ".tans"


def _default_output(args: argparse.Namespace) -> str:
    if args.input == "-":
        return "-"
    if args.command == "compress":
        return args.input + SUFFIX
    if args.input.endswith(SUFFIX):
        return args.input[: -len(SUFFIX)]
    raise SystemExit(
        f"pytans: cannot derive an output name from {args.input!r}; pass --output"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pytans",
        description="Compress and decompress files with the tANS entropy coder.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("compress", f"compress a file (default output: INPUT{SUFFIX})"),
        ("decompress", f"decompress a {SUFFIX} file"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("input", help="input file, or - for stdin")
        cmd.add_argument("-o", "--output", help="output file, or - for stdout")
        cmd.add_argument("-f", "--force", action="store_true", help="overwrite existing output")
        if name == "compress":
            cmd.add_argument(
                "--table-log",
                type=int,
                choices=range(MIN_TABLE_LOG, MAX_TABLE_LOG + 1),
                metavar=f"[{MIN_TABLE_LOG}-{MAX_TABLE_LOG}]",
                help=f"state table size as a power of two (default: auto, "
                f"at most {DEFAULT_MAX_TABLE_LOG})",
            )
            cmd.add_argument(
                "--block-size",
                type=int,
                default=DEFAULT_BLOCK_SIZE,
                metavar="BYTES",
                help=f"uncompressed bytes per block (default: {DEFAULT_BLOCK_SIZE})",
            )

    args = parser.parse_args(argv)
    output = args.output or _default_output(args)
    if output != "-" and Path(output).exists() and not args.force:
        raise SystemExit(f"pytans: refusing to overwrite {output} (use --force)")

    src = sys.stdin.buffer if args.input == "-" else open(args.input, "rb")
    made_output_file = output != "-"
    dst = sys.stdout.buffer if output == "-" else open(output, "wb")
    try:
        if args.command == "compress":
            if args.block_size < 1:
                raise SystemExit("pytans: --block-size must be at least 1")
            compress_stream(
                src, dst, block_size=args.block_size, table_log=args.table_log
            )
        else:
            decompress_stream(src, dst)
    except TansError as exc:
        if made_output_file:
            dst.close()
            Path(output).unlink(missing_ok=True)
        raise SystemExit(f"pytans: {exc}")
    finally:
        if src is not sys.stdin.buffer:
            src.close()
        if dst is not sys.stdout.buffer and not dst.closed:
            dst.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
