"""Evaluate a trained policy on the four conditions.

    a  matched       the instruction names the cube the scene was built around
    b  swapped       SAME initial layout, the instruction names a different cube,
                     and success is scored against the cube that was named
    c  no distractor only the target is on the bench
    d  occluded      the panel blocks the front camera for a window mid-episode

Condition (b) is the one that decides whether language is used at all. Each
layout is run once per colour, from the identical initial state, so a policy that
ignores the instruction and always pushes the same cube scores about 1/3 by
construction. For ACT, which has no language input, (b) is the identity
condition and that is the point of including it.

Every condition uses the same layouts, drawn from a seed block disjoint from the
training demonstrations, so the four numbers are paired.

Run:
    python scripts/evaluate_policy.py outputs/act_scripted_v1/checkpoints/last/pretrained_model \
        --n 60 --conditions a b c d
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.select_push import COLOURS, TEMPLATES, SelectPushEnv, lane_clear  # noqa: E402
from scripts.run_expert import classify, wilson  # noqa: E402

OCCLUDE_AT = 40
OCCLUDE_STEPS = 30


def load_policy(path: Path, device: str):
    """Load whatever policy type the checkpoint declares, with its own processors."""
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(str(path))
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(str(path), config=cfg)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(cfg, str(path))
    return policy, pre, post, cfg


def to_batch(obs: dict, device: str) -> dict:
    img = torch.from_numpy(obs["image"].copy()).permute(2, 0, 1).float().div(255.0)
    return {
        "observation.images.cam": img.unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(np.asarray(obs["state"], np.float32)).unsqueeze(0).to(device),
        "task": obs["instruction"],
    }


def run_episode(env: SelectPushEnv, policy, pre, post, device: str, reset_options: dict) -> dict:
    obs, _ = env.reset(seed=reset_options.pop("seed", None), options=reset_options)
    policy.reset()
    info: dict = {}
    n_occluded = 0
    while True:
        batch = pre(to_batch(obs, device))
        with torch.no_grad():
            action = policy.select_action(batch)
        action = post(action).squeeze(0).float().cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        n_occluded += int(info["occluded"])
        if terminated or truncated:
            break
    return {"success": bool(info["success"]), "outcome": classify(info, info["success"]),
            "target": env.target, "steps": env.t, "occluded_steps": n_occluded,
            "dist": float(info["target_goal_dist"])}


def layouts(n: int, seed0: int) -> list[dict]:
    """Sample n layouts once, so every condition sees the same scenes.

    Every colour's push lane must be clear, not just the target's. The env only
    guarantees the target's lane, so rotating the instruction in condition (b)
    could otherwise hand the named cube a blocked lane and turn a language test
    into a difficulty test.
    """
    src = SelectPushEnv()
    out, seed, tried = [], seed0, 0
    while len(out) < n:
        src.reset(seed=seed)
        lay = src.get_layout()
        if all(lane_clear(lay["cubes"], lay["goal"], c) for c in COLOURS):
            out.append({"seed": seed, "layout": lay, "target": src.target})
        seed += 1
        tried += 1
        if tried > 200 * n:
            raise RuntimeError(f"only {len(out)}/{n} all-lanes-clear layouts in {tried} seeds")
    print(f"{n} layouts with all three lanes clear, from seeds {seed0}..{seed - 1} "
          f"({n / tried:.0%} of seeds)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n", type=int, default=60, help="layouts per condition")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--conditions", nargs="+", default=["a", "b", "c", "d"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--template", type=int, default=0, help="instruction template index")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    a = ap.parse_args()

    policy, pre, post, cfg = load_policy(a.checkpoint, a.device)
    print(f"policy {cfg.type}, n_action_steps={cfg.n_action_steps}, chunk={cfg.chunk_size}")
    tpl = TEMPLATES[a.template]
    scenes = layouts(a.n, a.seed0)
    envs = {
        "a": SelectPushEnv(),
        "b": SelectPushEnv(),
        "c": SelectPushEnv(distractors=False),
        "d": SelectPushEnv(occlude_at=OCCLUDE_AT, occlude_steps=OCCLUDE_STEPS),
    }

    results: dict[str, list[dict]] = {}
    for cond in a.conditions:
        env = envs[cond]
        rows = []
        for i, sc in enumerate(scenes):
            if cond == "b":
                # same scene, instructed cube rotated to the next colour
                names = list(COLOURS)
                named = names[(names.index(sc["target"]) + 1) % len(names)]
            else:
                named = sc["target"]
            rows.append(run_episode(env, policy, pre, post, a.device, {
                "layout": sc["layout"], "target": named,
                "instruction": tpl.format(c=named)}) | {"seed": sc["seed"], "named": named})
            if (i + 1) % 20 == 0:
                k = sum(r["success"] for r in rows)
                print(f"  {cond}: {i + 1}/{a.n}  {k} ok", flush=True)
        results[cond] = rows
        k = sum(r["success"] for r in rows)
        lo, hi = wilson(k, len(rows))
        print(f"condition {cond}: {k}/{len(rows)}  ({k / len(rows):.0%}, "
              f"Wilson [{lo:.0%}, {hi:.0%}])  {dict(Counter(r['outcome'] for r in rows))}")

    a.out.mkdir(exist_ok=True)
    name = f"eval_{cfg.type}_{a.checkpoint.parent.parent.parent.name}.json"
    (a.out / name).write_text(json.dumps(
        {"checkpoint": str(a.checkpoint), "policy": cfg.type, "n": a.n, "seed0": a.seed0,
         "template": tpl, "results": results}, indent=1), encoding="utf-8")
    print(f"-> {a.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
