"""Minimal safetensors reader/writer (F16/F32 only, zero dependencies).

Format: 8-byte little-endian u64 header length N, then N bytes of JSON
({"name": {"dtype": "F16", "shape": [...], "data_offsets": [a, b]}, ...}),
then the raw tensor buffer. That is the entire spec for what we need.

The lazy reader mmaps the data buffer and materializes only requested rows,
which is what allows verifying a 4.5 TB sparse checkpoint in kilobytes of RAM.
"""
import json
import mmap
import os
import struct

import numpy as np

_DT = {"F16": np.float16, "F32": np.float32}
_DT_INV = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}


def _header_bytes(meta):
    h = json.dumps(meta, separators=(",", ":")).encode()
    pad = (-len(h)) % 8
    return h + b" " * pad


def save(path, tensors):
    """tensors: dict name -> numpy array (float16 or float32)."""
    meta, offset = {}, 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        meta[name] = {"dtype": _DT_INV[arr.dtype], "shape": list(arr.shape),
                      "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    h = _header_bytes(meta)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for arr in tensors.values():
            f.write(np.ascontiguousarray(arr).tobytes())


class LazyTensor:
    """mmap-backed lazy tensor. Only touched pages are read."""

    def __init__(self, path, name):
        self.path, self.name = path, name
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(n))[name]
        self.dtype = _DT[meta["dtype"]]
        self.shape = tuple(meta["shape"])
        a, b = meta["data_offsets"]
        self._off = 8 + n + a
        self.nbytes = b - a
        self._fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

    @property
    def numel(self):
        return int(np.prod(self.shape))

    def row(self, i):
        """Materialize a single row of a 2-D tensor (reads ~one row of pages)."""
        itemsize = np.dtype(self.dtype).itemsize
        start = self._off + i * self.shape[1] * itemsize
        return np.frombuffer(self._mm, dtype=self.dtype,
                             count=self.shape[1], offset=start)

    def close(self):
        self._mm.close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load(path, name):
    with LazyTensor(path, name) as t:
        itemsize = np.dtype(t.dtype).itemsize
        return np.frombuffer(t._mm, dtype=t.dtype, count=t.numel,
                             offset=t._off).reshape(t.shape).copy()
