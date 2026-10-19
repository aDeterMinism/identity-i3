"""identity_i3: the Identity I3 software development kit.

Loads the 2.25-trillion-parameter Identity I3 operator from its 3 MB
diagonal checkpoint and runs inference on commodity hardware.

    from identity_i3 import IdentityI3
    model = IdentityI3.from_pretrained("checkpoints/i3_diag_f16.safetensors")
    y = model(x)                      # y == x, up to float16 resolution
    print(model.num_parameters())     # 2250000000000
"""
import json
import os

import numpy as np


class IdentityI3:
    def __init__(self, diag):
        self._diag = np.asarray(diag)
        self.m = int(self._diag.shape[0])

    @classmethod
    def from_pretrained(cls, path):
        import st_io
        with st_io.LazyTensor(path, "diag") as t:
            diag = np.frombuffer(t._mm, dtype=t.dtype, count=t.numel,
                                 offset=t._off).copy()
        return cls(diag)

    def num_parameters(self):
        """Number of parameters of the represented linear operator R^m -> R^m."""
        return self.m * self.m

    def __call__(self, x):
        return self._diag.astype(np.float32) * np.asarray(x, dtype=np.float32)

    def __repr__(self):
        return (f"IdentityI3(m={self.m}, "
                f"parameters={self.num_parameters():.3e})")


def load_evidence_checkpoint(shard_paths):
    """Load a sharded evidence-run checkpoint (e.g. artifacts/m4096-0000{1,2}-of-00002).

    Returns the trained W as a float32 numpy array. Expect the diagonal near
    (1 - neg_frac) * c and everything else near zero (Proposition 2).
    """
    import st_io
    import numpy as np
    parts = [st_io.load(p, "weight") for p in shard_paths]
    return np.concatenate(parts, axis=0).astype(np.float32)
