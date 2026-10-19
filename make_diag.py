"""Generate the distributed checkpoint: the 2.25T operator in diagonal (DIA) format.

m F16 values = 3 MB. This is an exact, standard sparse matrix representation
of the same operator that make_weights.py materializes as a 4.5 TB dense file.
"""
import argparse

import numpy as np

import st_io

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=1_500_000)
ap.add_argument("--c", type=float, default=1.0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

diag = np.full(args.m, args.c, dtype=np.float16)
st_io.save(args.out, {"diag": diag})
print(f"wrote {args.out}: m={args.m}, operator parameters={args.m**2:.3e}, "
      f"file={2 * args.m} bytes payload")
