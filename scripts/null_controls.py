"""Null controls for the success criterion.

A success criterion that a random policy can satisfy, or that credits the wrong
cube, would make every later number meaningless. Four conditions on the same
seeds separate a real criterion from a broken one:

    random actions                 should be ~0
    zero actions                   should be 0
    expert pushing a distractor    should be 0   (criterion is bound to the target)
    expert pushing the target      should be ~all

Seeds 5000.. are disjoint from the Day 2 gate seeds (1000..1099), so the last row
is also an out-of-sample check on the expert.

Run:  python scripts/null_controls.py --n 50
Appends a table to docs/day2_expert.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.expert import PushExpert  # noqa: E402
from envs.select_push import COLOURS, SelectPushEnv  # noqa: E402

SEED0 = 5000


class Random:
    def __init__(self, env, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self):
        return self.rng.uniform(-1, 1, 3).astype(np.float32)


class Zero:
    def __init__(self, env):
        pass

    def reset(self):
        pass

    def act(self):
        return np.zeros(3, np.float32)


class Target:
    def __init__(self, env):
        self.ex = PushExpert(env)

    def reset(self):
        self.ex.reset()

    def act(self):
        return self.ex.act()


class Distractor:
    """The same expert, aimed at a cube that was not named in the instruction."""

    def __init__(self, env):
        self.env = env
        self.ex = PushExpert(env)

    def reset(self):
        self.ex.reset()
        self.true = self.env.target
        self.decoy = next(c for c in COLOURS if c != self.true)

    def act(self):
        self.env.target = self.decoy     # the expert reads env.target
        a = self.ex.act()
        self.env.target = self.true      # restored before the env scores the step
        return a


def run(env, policy, n: int) -> int:
    k = 0
    for ep in range(n):
        env.reset(seed=SEED0 + ep)
        policy.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy.act())
            done = term or trunc
        k += int(info["success"])
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    env = SelectPushEnv()
    rows = [
        ("random actions", Random(env), "~0"),
        ("zero actions", Zero(env), "0"),
        ("expert pushing a distractor", Distractor(env), "0"),
        ("expert pushing the instructed cube", Target(env), "~all"),
    ]
    results = []
    for label, pol, expect in rows:
        k = run(env, pol, args.n)
        results.append((label, k, expect))
        print(f"  {label:36s} {k:3d} / {args.n}   (expect {expect})")

    ok = (results[0][1] <= 1 and results[1][1] == 0 and results[2][1] == 0
          and results[3][1] >= 0.9 * args.n)
    print("criterion:", "VALID" if ok else "SUSPECT")

    with (REPO / "docs" / "day2_expert.md").open("a", encoding="utf-8") as f:
        f.write("\n## Null controls on the success criterion\n\n")
        f.write(f"Seeds {SEED0}..{SEED0 + args.n - 1}, disjoint from the gate seeds, "
                f"produced by `scripts/null_controls.py --n {args.n}`.\n\n")
        f.write("| policy | success | expected |\n|---|---|---|\n")
        for label, k, expect in results:
            f.write(f"| {label} | {k}/{args.n} | {expect} |\n")
        f.write(f"\nCriterion **{'valid' if ok else 'suspect'}**: it is not satisfied by "
                "chance, and it is bound to the instructed cube rather than to any cube "
                "reaching the goal.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
