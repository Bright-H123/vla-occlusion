"""Record episodes of SelectPushEnv into a LeRobotDataset.

One recorder is used by every data source, so that a scripted set and a
human-teleoperated set differ only in where the action comes from. The source is
written into the dataset card and into a per-episode sidecar, never inferred.

Policy inputs and logging fields are deliberately in different namespaces.
LeRobot's `dataset_to_policy_features` turns every key starting with
"observation" into a policy STATE input and every key starting with "action"
into an ACTION, so anything recorded for bookkeeping (seed, layout, source,
whether the frame was occluded) would silently become a network input if it were
named `observation.*`. Those fields go to a sidecar JSON instead, keyed by
episode index, and the dataset itself carries only:

    observation.images.cam   uint8 (H, W, 3)
    observation.state        float32 (4,)   TCP xyz and the contact flag
    action                   float32 (3,)   end-effector delta, in [-1, 1]
    task                     str            the instruction

Windows note: save_episode encodes video in a process pool, so any script that
calls this module must be under `if __name__ == "__main__":`.

Run:  python data/record.py --n 180 --out data/scripted_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.expert import PushExpert  # noqa: E402
from envs.select_push import CAM_RES, CONTROL_HZ, SelectPushEnv  # noqa: E402

CAM_KEY = "observation.images.cam"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def features(resolution: int = CAM_RES) -> dict:
    return {
        CAM_KEY: {"dtype": "video", "shape": (resolution, resolution, 3),
                  "names": ["height", "width", "channel"]},
        STATE_KEY: {"dtype": "float32", "shape": (4,),
                    "names": ["tcp_x", "tcp_y", "tcp_z", "contact"]},
        ACTION_KEY: {"dtype": "float32", "shape": (3,),
                     "names": ["dx", "dy", "dz"]},
    }


@dataclass
class EpisodeRecord:
    """Everything about an episode that is NOT a policy input."""
    episode_index: int
    seed: int
    source: str
    target: str
    instruction: str
    success: bool
    steps: int
    layout: dict
    occluded_steps: int
    masked: bool


class Recorder:
    """Wraps a LeRobotDataset and a sidecar, and keeps them in step."""

    def __init__(self, out: Path, source: str, fps: int = CONTROL_HZ,
                 resolution: int = CAM_RES, repo_id: str = "local/select_push"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.out = Path(out)
        self.source = source
        self.ds = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features(resolution),
            root=str(self.out), robot_type="franka_panda_pusher", use_videos=True,
        )
        self.records: list[EpisodeRecord] = []

    def add(self, obs: dict, action: np.ndarray) -> None:
        self.ds.add_frame({
            CAM_KEY: obs["image"],
            STATE_KEY: np.asarray(obs["state"], np.float32),
            ACTION_KEY: np.asarray(action, np.float32),
            "task": obs["instruction"],
        })

    def end_episode(self, rec_kwargs: dict) -> None:
        rec = EpisodeRecord(episode_index=len(self.records), source=self.source, **rec_kwargs)
        self.records.append(rec)
        self.ds.save_episode()

    def drop_episode(self) -> None:
        """Discard the frames buffered for the current episode without saving it."""
        self.ds.clear_episode_buffer()

    def close(self) -> None:
        self.ds.finalize()
        side = self.out / "episodes.jsonl"
        with side.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        card = self.out / "SOURCE.md"
        n_ok = sum(r.success for r in self.records)
        card.write_text(
            f"# Data source\n\nsource: **{self.source}**\n\n"
            f"episodes: {len(self.records)} ({n_ok} successful)\n\n"
            "`scripted` means the action column was produced by envs/expert.py from privileged\n"
            "simulator state. It is synthetic data and is not teleoperation.\n\n"
            "`human` means a person drove the end effector live through the teleoperation\n"
            "interface, and every action can be recomputed from the raw sensor packets stored\n"
            "alongside this dataset.\n\n"
            "Per-episode bookkeeping is in `episodes.jsonl`, deliberately outside the dataset\n"
            "features so that it cannot become a policy input.\n",
            encoding="utf-8")


def collect_scripted(out: Path, n: int, seed0: int, keep_failures: bool,
                     occlude_at: int | None, occlude_steps: int) -> None:
    env = SelectPushEnv(occlude_at=occlude_at, occlude_steps=occlude_steps)
    expert = PushExpert(env)
    rec = Recorder(out, source="scripted")
    t0 = time.perf_counter()
    kept = dropped = 0
    for i in range(n):
        seed = seed0 + i
        obs, _ = env.reset(seed=seed)
        layout = env.get_layout()
        expert.reset()
        n_occ = 0
        info = {}
        while True:
            action = expert.act()
            rec.add(obs, action)
            obs, _, terminated, truncated, info = env.step(action)
            n_occ += int(info["occluded"])
            if terminated or truncated:
                break
        if info["success"] or keep_failures:
            rec.end_episode(dict(seed=seed, target=env.target, instruction=env.instruction,
                                 success=bool(info["success"]), steps=env.t, layout=layout,
                                 occluded_steps=n_occ, masked=bool(info["masked"])))
            kept += 1
        else:
            rec.drop_episode()
            dropped += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  kept {kept}  dropped {dropped}", flush=True)
    rec.close()
    dt = time.perf_counter() - t0
    print(f"kept {kept} episodes, dropped {dropped}, {dt:.1f} s "
          f"({dt / max(n, 1):.2f} s/episode) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "scripted_v1")
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--keep-failures", action="store_true",
                    help="record unsuccessful episodes too; off by default")
    ap.add_argument("--occlude-at", type=int, default=None)
    ap.add_argument("--occlude-steps", type=int, default=30)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; remove it or choose another --out")
    collect_scripted(a.out, a.n, a.seed0, a.keep_failures, a.occlude_at, a.occlude_steps)


if __name__ == "__main__":
    main()
