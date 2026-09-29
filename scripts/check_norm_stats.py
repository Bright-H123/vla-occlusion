"""Check that a policy checkpoint carries normalisation statistics for THIS dataset.

A missing statistic does not raise: LeRobot's normaliser silently skips keys it
has no entry for, so the policy trains and runs on unnormalised state and
actions and simply performs badly. The released lerobot/smolvla_base normaliser,
for example, holds only dataset-prefixed keys (so100.buffer.action.*) and no plain
observation.state.* or action.* at all.

This script fails loudly instead. It requires the plain keys to be present and,
given the dataset, requires them to equal the dataset's own statistics.

Run:
    python scripts/check_norm_stats.py outputs/<run>/checkpoints/<step>/pretrained_model \
        --dataset-root data/scripted_v1
Exit code 0 means usable, 1 means the checkpoint must not be evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REQUIRED = ("observation.state.mean", "observation.state.std", "action.mean", "action.std")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="a pretrained_model directory")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--atol", type=float, default=1e-5)
    a = ap.parse_args()

    files = sorted(a.checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not files:
        print(f"FAIL  no normaliser file in {a.checkpoint}")
        return 1
    st = load_file(str(files[0]))
    print(f"normaliser: {files[0].name}, {len(st)} tensors")

    missing = [k for k in REQUIRED if k not in st]
    if missing:
        print(f"FAIL  missing {missing}")
        print(f"      keys present: {sorted(st)}")
        return 1

    if a.dataset_root is not None:
        ds = json.loads((a.dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))
        for key in REQUIRED:
            feat, stat = key.rsplit(".", 1)
            want = np.asarray(ds[feat][stat], np.float32).ravel()
            got = st[key].astype(np.float32).ravel()
            if want.shape != got.shape or not np.allclose(want, got, atol=a.atol):
                print(f"FAIL  {key} differs from the dataset\n      ckpt {got}\n      data {want}")
                return 1
        print(f"stats match {a.dataset_root}")

    for key in REQUIRED:
        print(f"  {key:26s} {np.round(st[key].ravel(), 4).tolist()}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
