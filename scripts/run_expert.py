"""Day 2 gate: run the scripted expert on N random initial states.

The gate is 90 successes out of 100. Below 80 the plan says stop tuning physics
and fall back (see the kill criteria), so the exact count matters and is written
into docs/day2_expert.md.

Run:  python scripts/run_expert.py --n 100 --gif 3 --record

Without --record nothing is written. The record is rewritten as a whole and
null_controls.py appends to it, so a quick check such as --n 20 must not be
allowed to replace the 100-episode gate and drop the null-control table.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.expert import PushExpert  # noqa: E402
from envs.select_push import SelectPushEnv  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def classify(info: dict, success: bool) -> str:
    if success:
        return "success"
    if info["wrong_cube_on_goal"]:
        return "wrong cube on goal"
    if info["out_of_bounds"]:
        return "cube pushed out of bounds"
    if info["target_goal_dist"] < 0.08:
        return "near miss (<8 cm)"
    return "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--gif", type=int, default=0, help="save GIFs of the first K episodes")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="rewrite docs/day2_expert.md; only for the full gate run")
    args = ap.parse_args()

    env = SelectPushEnv()
    expert = PushExpert(env)
    outcomes, steps, restages = Counter(), [], []
    t0 = time.perf_counter()

    for ep in range(args.n):
        obs, info = env.reset(seed=args.seed0 + ep)
        expert.reset()
        frames = [obs["image"]] if ep < args.gif else None
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(expert.act())
            done = term or trunc
            if frames is not None:
                frames.append(obs["image"])
        success = bool(info["success"])
        tag = classify(info, success)
        outcomes[tag] += 1
        steps.append(env.t)
        restages.append(expert.restages)
        if not args.quiet:
            print(f"ep {ep:3d}  seed {args.seed0 + ep}  {info['target']:5s}  "
                  f"{tag:26s}  t={env.t:3d}  restage={expert.restages}")
        if frames is not None:
            import imageio.v3 as iio
            out = REPO / "docs" / f"day2_expert_ep{ep}_{tag.split()[0]}.gif"
            iio.imwrite(out, np.stack(frames), duration=1000 // 20, loop=0)

    k = outcomes["success"]
    lo, hi = wilson(k, args.n)
    dt = time.perf_counter() - t0
    ok_steps = [s for s, _ in zip(steps, range(args.n))]
    print("\n" + "=" * 60)
    print(f"expert success: {k}/{args.n}  ({k / args.n:.0%}, Wilson 95% [{lo:.0%}, {hi:.0%}])")
    for tag, c in outcomes.most_common():
        print(f"  {tag:28s} {c}")
    print(f"mean episode length {np.mean(steps):.0f} steps, mean re-stages {np.mean(restages):.2f}")
    print(f"wall clock {dt:.1f} s for {args.n} episodes ({dt / args.n:.2f} s/ep)")
    gate = "PASS" if k >= 90 * args.n / 100 else ("FALLBACK" if k < 80 * args.n / 100 else "BELOW TARGET")
    print(f"Day 2 gate (>=90/100): {gate}")

    if not args.record:
        return 0
    (REPO / "docs" / "day2_expert.md").write_text(
        "\n".join([
            "# Day 2 — scripted expert gate",
            "",
            f"Seeds {args.seed0}..{args.seed0 + args.n - 1}, one episode each, "
            f"produced by `scripts/run_expert.py --n {args.n}`.",
            "",
            f"**Success: {k}/{args.n}** ({k / args.n:.0%}, Wilson 95% interval "
            f"[{lo:.0%}, {hi:.0%}]). Gate is 90/100. **{gate}.**",
            "",
            "| outcome | count |",
            "|---|---|",
            *[f"| {t} | {c} |" for t, c in outcomes.most_common()],
            "",
            f"Mean episode length {np.mean(steps):.0f} control steps (20 Hz). "
            f"Mean re-stages per episode {np.mean(restages):.2f}. "
            f"{dt / args.n:.2f} s wall clock per episode.",
            "",
        ]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
