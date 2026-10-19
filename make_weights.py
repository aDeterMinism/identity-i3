"""Materialize the full 2.25T-parameter Identity I3 checkpoint as a sparse file.

The resulting file is a byte-level valid dense safetensors checkpoint of shape
(m, m), F16, logical size ~4.5 TB, in which exactly m entries (the diagonal)
are nonzero. On a filesystem that supports sparse files (ext4, xfs, APFS)
the zeros are never stored: physical usage is ~m pages = a few GB.

Distribution model: you do not download this file. You regenerate it, locally,
in minutes, from this script. The bytes you obtain are identical to ours.

    python3 make_weights.py --m 1500000 --c 1.0 --out i3_2T.safetensors

Requires Linux or macOS. Verification: see verify.py.
"""
import argparse
import json
import os
import struct
import time

import numpy as np

CHUNK = 65536  # rows per progress report


def f16_bytes(val):
    return np.float16(val).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1_500_000)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    m = args.m
    data_nbytes = 2 * m * m  # F16
    meta = {"weight": {"dtype": "F16", "shape": [m, m],
                       "data_offsets": [0, data_nbytes]}}
    h = json.dumps(meta, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    off0 = 8 + len(h)
    total = off0 + data_nbytes

    print(f"logical size : {total / 2**40:.2f} TiB ({total} bytes)")
    print(f"parameters   : {m * m:.3e}")
    t0 = time.time()

    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, total)
    os.pwrite(fd, struct.pack("<Q", len(h)) + h, 0)

    two = f16_bytes(args.c)
    row = 2 * m + 2  # stride between diagonal entries, in bytes
    for i in range(m):
        os.pwrite(fd, two, off0 + i * row)
        if (i + 1) % CHUNK == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{m} rows  ({dt:.1f}s, "
                  f"{(i + 1) / dt:.0f} rows/s)", flush=True)
    os.close(fd)

    st = os.stat(args.out)
    print(f"done in {time.time() - t0:.1f}s")
    print(f"physical     : {st.st_blocks * 512 / 2**30:.2f} GiB "
          f"(du-style block count)")
    print(f"compression  : logical/physical = "
          f"{total / max(st.st_blocks * 512, 1):.0f}x")


if __name__ == "__main__":
    main()
