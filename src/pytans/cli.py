"""Command-line interface: ``pytans compress`` / ``pytans decompress``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, compress, decompress
from .exceptions import TansError
from .tables import DEFAULT_MAX_TABLE_LOG, MAX_TABLE_LOG, MIN_TABLE_LOG

SUFFIX = ".tans"


def _read_input(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def _write_output(path: str, data: bytes, force: bool) -> None:
    if path == "-":
        sys.stdout.buffer.write(data)
        return
    target = Path(path)
    if target.exists() and not force:
        raise SystemExit(f"pytans: refusing to overwrite {target} (use --force)")
    target.write_bytes(data)


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

    args = parser.parse_args(argv)
    output = args.output or _default_output(args)
    data = _read_input(args.input)
    try:
        if args.command == "compress":
            result = compress(data, table_log=args.table_log)
        else:
            result = decompress(data)
    except TansError as exc:
        raise SystemExit(f"pytans: {exc}")
    _write_output(output, result, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
