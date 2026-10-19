"""Verify Identity I3 claims. Each subcommand checks one row of the model card.

  python3 verify.py params  --m 1500000
      Parameter count of the m x m operator, via a meta-device nn.Linear:
      the parameters exist as bona fide torch.nn.Parameter objects at zero
      memory cost. This is the number the draft regulation counts.

  python3 verify.py forward --ckpt checkpoints/i3_diag_f16.safetensors
      Full forward pass of the 2.25T model on this machine, from the 3 MB
      diagonal checkpoint: y = diag . x. Runs in well under a second.

  python3 verify.py sparse --file i3_2T.safetensors [--rows 16]
      Verify a regenerated 4.5 TB sparse checkpoint: header validity, lazily
      materialize random rows and check they equal c * e_i, then enumerate
      data blocks with SEEK_DATA/SEEK_HOLE.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

import st_io


def cmd_params(args):
    import torch
    layer = torch.nn.Linear(args.m, args.m, bias=False, device="meta")
    n = sum(p.numel() for p in layer.parameters())
    print(f"nn.Linear({args.m}, {args.m}) on meta device")
    print(f"sum(p.numel() for p in parameters()) = {n:.6e}")
    print(f"draft parameter threshold            = 2.000e+12")
    print(f"threshold crossed: {n >= 2e12}")


def cmd_forward(args):
    with st_io.LazyTensor(args.ckpt, "diag") as t:
        diag = np.frombuffer(t._mm, dtype=t.dtype, count=t.numel,
                             offset=t._off).copy()
    m = diag.shape[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(m).astype(np.float32)
    y = diag.astype(np.float32) * x
    err = float(np.max(np.abs(y - x * diag.astype(np.float32).mean())))
    print(f"checkpoint : {args.ckpt}")
    print(f"m          : {m}  (operator parameters: {m * m:.3e})")
    print(f"forward    : y = diag . x for random x in R^{m} -- done")
    print(f"max |y - c.x| where c = mean(diag): {err:.3e}")


def cmd_sparse(args):
    import torch  # only for nn.Parameter numel cross-check
    with st_io.LazyTensor(args.file, "weight") as t:
        m = t.shape[0]
        print(f"file       : {args.file}")
        print(f"shape      : {t.shape}, dtype {t.dtype.__name__}")
        print(f"numel      : {t.numel:.6e}")
        st = os.stat(args.file)
        print(f"logical    : {st.st_size / 2**40:.2f} TiB")
        print(f"physical   : {st.st_blocks * 512 / 2**30:.2f} GiB")

        rng = np.random.default_rng(42)
        idx = rng.integers(0, m, size=args.rows)
        for i in idx:
            row = t.row(int(i)).astype(np.float32)
            nz = np.nonzero(row)[0]
            ok = len(nz) == 1 and nz[0] == i
            val = row[i] if len(nz) else float("nan")
            print(f"row {i:>8}: nnz={len(nz)}, diag value={val:.3f}, "
                  f"{'OK' if ok else 'FAIL'}")

    fd = os.open(args.file, os.O_RDONLY)
    size = os.path.getsize(args.file)
    pos, blocks, data_bytes = 0, 0, 0
    while pos < size:
        try:
            d = os.lseek(fd, pos, os.SEEK_DATA)
        except OSError:
            break
        e = os.lseek(fd, d, os.SEEK_HOLE)
        blocks += 1
        data_bytes += e - d
        pos = e
    os.close(fd)
    print(f"SEEK_DATA extents: {blocks}, covering {data_bytes / 2**30:.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("params"); p.add_argument("--m", type=int, default=1_500_000)
    p = sub.add_parser("forward"); p.add_argument("--ckpt", required=True)
    p = sub.add_parser("sparse"); p.add_argument("--file", required=True)
    p.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()
    {"params": cmd_params, "forward": cmd_forward, "sparse": cmd_sparse}[args.cmd](args)
