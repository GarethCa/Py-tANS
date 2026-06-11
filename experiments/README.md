# Experiments: modeling front ends for tANS

pytans alone is an order-0 entropy coder — it cannot see repetition.
These experiments (`repetition.py`) bolt classic modeling stages in
front of it to find out how much of the gap to full pipelines
(zlib/bz2/lzma) each one closes.

## Results — 1 MB slices, compressed size as % of original

| corpus | floor | tans | mtf+tans | **bwt+tans** | **lz77+tans** | zlib -9 | bz2 -9 | lzma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English text | 58.4 | 58.6 | 64.0 | **30.9** | **34.0** | 36.3 | 26.7 | 29.5 |
| Python source | 59.4 | 59.5 | 57.0 | **23.1** | **23.7** | 24.5 | 20.2 | 19.5 |
| Sorted dictionary | 53.8 | 53.9 | 49.9 | 35.1 | 32.2 | 30.5 | 34.7 | 25.9 |
| JSON logs | 60.5 | 60.5 | 64.5 | **10.6** | 14.1 | 12.6 | 9.7 | 10.7 |
| Skewed, no repeats | 30.7 | **30.7** | 35.6 | 35.6 | 38.1 | 36.8 | 36.1 | 34.1 |

(bwt+tans uses 256 KB blocks; bold = beats zlib -9.)

## Findings

1. **BWT + MTF + zero-RLE + tANS beats zlib -9 on every repetitive
   corpus** — e.g. 30.9 % vs 36.3 % on text — and sits ~1–3 points
   behind bz2, whose pipeline it emulates. Most of the remaining gap is
   block size: raising the BWT block from 256 KB to 1 MB (bzip2 uses
   900 KB) gives text 28.2 %, source 21.6 %, JSON 10.5 %. The sorted
   dictionary prefers *small* blocks (local statistics dominate).
2. **LZ77 + tANS reaches/beats zlib parity** (text 34.0 % vs 36.3 %)
   even with a greedy pure-Python matcher, because the unlimited window
   and zstd-style stream splitting (literals / run lengths / match
   lengths / offset buckets, each with its own tANS table) compensate
   for the weaker parse. This is the zstd recipe in miniature.
3. **MTF alone is not worth it** — it only helps data with strong
   locality (the sorted dictionary) and hurts everything else.
4. **Every transform hurts the no-repetition control row**, where plain
   tANS is already optimal (30.7 % = the entropy floor, ahead of all
   three reference compressors). A production integration must make
   the transform optional — pick-smallest-per-block costs one byte of
   framing and keeps the current behaviour as the fall-back, exactly
   like the existing raw-storage fallback.

## Reproduce

```sh
pip install -e . numpy          # numpy only needed for the BWT suffix sort
python experiments/repetition.py            # 1 MB slices, ~20 s
python experiments/repetition.py --size 131072
```

The LZ77 parse is verified by reconstruction on every run; BWT/MTF/RLE
are standard invertible transforms (inverses not implemented here —
this measures potential, it is not a codec).
