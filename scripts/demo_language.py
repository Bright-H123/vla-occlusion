"""Same scene, three instructions.

Finds a layout in which every cube has a clear push corridor, then runs the
scripted expert three times from that identical initial state, once per
instruction. The three panels differ only in the instruction.

This is the demonstration of what the project is about, and the exact mechanism
the swapped-instruction evaluation relies on: reset(options={"layout": ...}).

Run:  python scripts/demo_language.py
Writes docs/demo_same_scene_three_instructions.gif and a still PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.expert import PushExpert  # noqa: E402
from envs.select_push import COLOURS, TEMPLATES, SelectPushEnv, corridor_clear  # noqa: E402

RES = 224
CAPTION_H = 46
RETRACT = [(0, 0, 1)] * 8 + [(-1, 0, 0.3)] * 14


def caption(img: np.ndarray, text: str, ok: bool | None) -> np.ndarray:
    bar = np.full((CAPTION_H, img.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, '"' + text + '"', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (30, 30, 30), 1, cv2.LINE_AA)
    if ok is not None:
        colour = (40, 150, 60) if ok else (200, 50, 50)
        cv2.putText(bar, "success" if ok else "fail", (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def find_shared_layout(env: SelectPushEnv, seed0: int = 2000) -> tuple[int, dict]:
    for s in range(seed0, seed0 + 2000):
        env.reset(seed=s, options={"target": "red"})
        lay = env.get_layout()
        if all(corridor_clear(lay["cubes"], lay["goal"], t) for t in COLOURS):
            return s, lay
    raise RuntimeError("no layout with three clear corridors in 2000 seeds")


def main() -> int:
    env = SelectPushEnv(resolution=RES)
    expert = PushExpert(env)
    seed, layout = find_shared_layout(env)
    print(f"shared layout from seed {seed}")

    runs = []
    for colour in COLOURS:
        text = TEMPLATES[0].format(c=colour)
        obs, _ = env.reset(seed=seed, options={"layout": layout, "target": colour,
                                               "instruction": text})
        expert.reset()
        frames, done = [obs["image"]], False
        while not done:
            obs, _, term, trunc, info = env.step(expert.act())
            done = term or trunc
            frames.append(obs["image"])
        ok = bool(info["success"])   # scored before the arm moves away
        ep_steps = env.t
        # after the episode, lift and pull the arm back so the result is visible;
        # this is presentation only and does not affect the success flag above
        for a in RETRACT:
            obs, *_ = env.step(np.array(a, np.float32))
            frames.append(obs["image"])
        runs.append((text, frames, ok))
        print(f"  {text:34s} success={ok}  steps={ep_steps}")

    # sanity: the three runs really started from the same state
    first = [r[1][0] for r in runs]
    same = all(np.array_equal(first[0], f) for f in first[1:])
    print("identical first frame across the three runs:", same)

    T = max(len(f) for _, f, _ in runs) + 20
    gif = []
    for t in range(T):
        panels = []
        for text, frames, ok in runs:
            img = frames[min(t, len(frames) - 1)]
            panels.append(caption(img, text, ok if t >= len(frames) - len(RETRACT) - 1 else None))
        gif.append(np.concatenate(panels, 1))
    out = REPO / "docs" / "demo_same_scene_three_instructions.gif"
    iio.imwrite(out, np.stack(gif), duration=50, loop=0)

    still = np.concatenate([np.concatenate([caption(f[0], t, None), caption(f[-1], t, ok)], 0)
                            for t, f, ok in runs], 1)
    iio.imwrite(REPO / "docs" / "demo_same_scene_three_instructions.png", still)
    print(f"wrote {out.name} ({len(gif)} frames) and the still PNG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
