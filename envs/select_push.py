"""Language-conditioned selective pushing, a Gymnasium environment.

A Franka Panda with a cylindrical pusher on its hand must push the cube named in
a natural-language instruction onto a goal pad, with two same-sized cubes of other
colours present as distractors.

Scene construction is done in code with MjSpec on top of the unmodified
mujoco_menagerie Panda, so the upstream model file is never copied or edited.

Control is end-effector position control. Each control step, damped least-squares
differential IK on a scratch copy of the state produces joint targets, which are
sent to the Panda's own position actuators. There is no mocap body and no weld
constraint: the arm is driven through its real actuators.

Observation (dict):
    image        uint8 (H, W, 3)   front camera by default (see _build_model)
    state        float32 (4,)      TCP x, y, z and pusher-contact flag
    instruction  str               natural-language command

Action: float32 (3,) in [-1, 1], an end-effector displacement (dx, dy, dz)
scaled by MAX_STEP metres per control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PANDA_XML = REPO / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

# --------------------------------------------------------------------------- task
COLOURS = {
    "red":   (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.70, 0.20, 1.0),
    "blue":  (0.15, 0.35, 0.85, 1.0),
}
MASK_RGBA = (0.55, 0.55, 0.55, 1.0)  # used when the colour cue is removed

TEMPLATES = [
    "push the {c} cube to the goal",
    "move the {c} block onto the target",
    "slide the {c} cube onto the pad",
]

CUBE_HALF = 0.02
PUSHER_R = 0.015
GOAL_R = 0.04

# workspace for object placement, in the robot base frame (metres)
WS_X = (0.45, 0.66)
WS_Y = (-0.17, 0.17)
MIN_OBJ_SEP = 0.10     # between cube centres
MIN_GOAL_SEP = 0.12    # goal centre to any cube centre
CORRIDOR_HALF = 0.055  # clear half-width around the target's push line

# -------------------------------------------------------------------------- control
CONTROL_HZ = 20
MAX_STEP = 0.012       # m per control step at |a| = 1, i.e. 0.24 m/s
TCP_OFFSET = 0.165     # hand origin to pusher tip, along the hand z-axis
TCP_Z_MIN, TCP_Z_MAX = 0.008, 0.20
TCP_X = (0.35, 0.75)
TCP_Y = (-0.28, 0.28)

IK_ITERS = 12
IK_DAMPING = 1e-2

EPISODE_STEPS = 160    # 8 s at 20 Hz
SETTLE_STEPS = 10      # target must sit on the goal this long to count
SETTLE_SPEED = 0.01    # m/s

CAM_RES = 128


@dataclass
class Info:
    target: str
    success: bool
    target_goal_dist: float
    wrong_cube_on_goal: bool
    out_of_bounds: bool
    masked: bool


def _build_model() -> mujoco.MjModel:
    """Panda + pusher + bench + three cubes + goal pad + two cameras."""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.option.timestep = 0.002
    wb = spec.worldbody

    # The real Franka controller compensates gravity; the Menagerie model does not,
    # so a pure P position servo sags 4-9 mm below its target. Turn it on.
    for b in spec.bodies:
        if b.name.startswith(("link", "hand", "left_finger", "right_finger")):
            b.gravcomp = 1.0

    # panda.xml ships a COM-tracking spotlight that casts large arm shadows across
    # the bench. Those shadows move with the arm and add a strong visual cue that
    # has nothing to do with the task, so keep the light but drop its shadow.
    for lt in spec.lights:
        lt.castshadow = False

    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]

    wb.add_light(pos=[0.55, 0.0, 1.5], dir=[0, 0, -1], diffuse=[0.55, 0.55, 0.55],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2, 2, 0.05], rgba=[0.16, 0.16, 0.16, 1],
                friction=[0.6, 0.005, 0.0001])
    # bench mat, visual only, dark so that the goal pad and the cubes stand out
    wb.add_geom(name="bench", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.555, 0.0, 0.0005], size=[0.16, 0.24, 0.0005],
                rgba=[0.20, 0.27, 0.20, 1], contype=0, conaffinity=0)

    # goal pad, visual only, in a colour no cube uses
    goal = wb.add_body(name="goal", pos=[0.55, 0.0, 0.0012], mocap=True)
    goal.add_geom(name="goal_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[GOAL_R, 0.0008, 0], rgba=[0.96, 0.94, 0.86, 1],
                  contype=0, conaffinity=0)

    for i, (name, rgba) in enumerate(COLOURS.items()):
        b = wb.add_body(name=f"cube_{name}", pos=[0.5, -0.1 + 0.1 * i, CUBE_HALF])
        b.add_freejoint(name=f"cube_{name}_free")
        b.add_geom(name=f"cube_{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[CUBE_HALF] * 3, rgba=list(rgba), mass=0.06,
                   friction=[0.6, 0.005, 0.0001])

    # pusher on the hand; its tip is the controlled point (TCP)
    hand = spec.body("hand")
    hand.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, TCP_OFFSET - 0.035], size=[PUSHER_R, 0.035, 0],
                  rgba=[0.9, 0.9, 0.9, 1], mass=0.02,
                  friction=[0.3, 0.005, 0.0001])
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0])

    wb.add_camera(name="top", pos=[0.585, 0.0, 0.78],
                  xyaxes=[0, -1, 0, 1, 0, 0], fovy=40)
    # front camera on the far side of the bench, looking back at the robot, so the
    # arm enters from the far edge instead of hovering over the objects it pushes
    wb.add_camera(name="front", pos=[0.98, 0.0, 0.52],
                  xyaxes=[0, 1, 0, -0.774, 0, 0.633], fovy=36)
    return spec.compile()


class SelectPushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode: str | None = "rgb_array",
                 distractors: bool = True, mask_colour_at: int | None = None,
                 resolution: int = CAM_RES, camera: str = "front"):
        super().__init__()
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.distractors = distractors
        self.mask_colour_at = mask_colour_at
        self.res = resolution
        self.camera = camera
        self._renderer = mujoco.Renderer(self.model, height=resolution, width=resolution)

        m = self.model
        self.n_sub = int(round(1.0 / (CONTROL_HZ * m.opt.timestep)))
        self._hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._pusher = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pusher")
        self._goal_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal")]
        self._arm_dofs = np.arange(7)
        self._jrange = m.jnt_range[:7].copy()

        self._cube = {}
        for name in COLOURS:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cube_{name}")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{name}_free")
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{name}_geom")
            self._cube[name] = (bid, m.jnt_qposadr[jid], m.jnt_dofadr[jid], gid)

        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)
        self._home_q = self.data.qpos[:7].copy()
        self._R_target = self.data.xmat[self._hand].reshape(3, 3).copy()

        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (resolution, resolution, 3), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "instruction": gym.spaces.Text(max_length=64),
        })

    # ------------------------------------------------------------------ helpers
    def cube_xy(self, name: str) -> np.ndarray:
        return self.data.xpos[self._cube[name][0]][:2].copy()

    def cube_speed(self, name: str) -> float:
        dof = self._cube[name][2]
        return float(np.linalg.norm(self.data.qvel[dof:dof + 2]))

    def goal_xy(self) -> np.ndarray:
        return self.data.mocap_pos[self._goal_mocap][:2].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp].copy()

    def _pusher_in_contact(self) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            if self._pusher in (c.geom1, c.geom2):
                return True
        return False

    def _set_cube(self, name: str, xy: np.ndarray, yaw: float) -> None:
        _, qadr, dadr, _ = self._cube[name]
        self.data.qpos[qadr:qadr + 3] = [xy[0], xy[1], CUBE_HALF]
        self.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[dadr:dadr + 6] = 0

    def _park_cube(self, name: str) -> None:
        """Move a cube well outside the camera view, used for the no-distractor condition."""
        self._set_cube(name, np.array([1.5, 1.5 + list(COLOURS).index(name) * 0.2]), 0.0)

    def _ik(self, target_pos: np.ndarray) -> np.ndarray:
        """Damped least-squares IK for position plus a fixed downward orientation."""
        m, d = self.model, self._scratch
        d.qpos[:] = self.data.qpos
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(IK_ITERS):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self._tcp]
            R = d.xmat[self._hand].reshape(3, 3)
            e_pos = target_pos - p
            e_rot = 0.5 * sum(np.cross(R[:, i], self._R_target[:, i]) for i in range(3))
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self._tcp)
            J = np.vstack([jacp[:, self._arm_dofs], jacr[:, self._arm_dofs]])
            e = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), e)
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, self._jrange[:, 0], self._jrange[:, 1])
        return d.qpos[:7].copy()

    def _place_objects(self) -> None:
        rng = self.np_random
        names = list(COLOURS)
        for _ in range(500):
            pts = [np.array([rng.uniform(*WS_X), rng.uniform(*WS_Y)]) for _ in names]
            if min(np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]) < MIN_OBJ_SEP:
                continue
            goal = np.array([rng.uniform(*WS_X), rng.uniform(*WS_Y)])
            if min(np.linalg.norm(goal - p) for p in pts) < MIN_GOAL_SEP:
                continue
            c = pts[names.index(self.target)]
            u = (goal - c) / np.linalg.norm(goal - c)
            pre = c - u * 0.07
            if not (WS_X[0] - 0.05 <= pre[0] <= WS_X[1] + 0.05 and TCP_Y[0] <= pre[1] <= TCP_Y[1]):
                continue
            # the target's push corridor, including the approach point, must be clear
            clear = True
            for n, p in zip(names, pts):
                if n == self.target:
                    continue
                t = np.clip(np.dot(p - pre, u), 0.0, np.linalg.norm(goal - pre))
                if np.linalg.norm(p - (pre + t * u)) < CORRIDOR_HALF + CUBE_HALF:
                    clear = False
                    break
            if clear:
                break
        else:
            raise RuntimeError("could not sample a valid layout in 500 tries")

        for n, p in zip(names, pts):
            self._set_cube(n, p, rng.uniform(-np.pi, np.pi))
        if not self.distractors:
            for n in names:
                if n != self.target:
                    self._park_cube(n)
        self.data.mocap_pos[self._goal_mocap][:2] = goal

    def _set_colours(self, masked: bool) -> None:
        for name, (_, _, _, gid) in self._cube.items():
            self.model.geom_rgba[gid] = MASK_RGBA if masked else COLOURS[name]

    def _obs(self) -> dict:
        self._renderer.update_scene(self.data, camera=self.camera)
        img = self._renderer.render()
        tcp = self.tcp_pos()
        state = np.array([tcp[0], tcp[1], tcp[2], float(self._pusher_in_contact())], np.float32)
        return {"image": img, "state": state, "instruction": self.instruction}

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)

        self.target = options.get("target") or self.np_random.choice(list(COLOURS))
        tpl = options.get("template")
        if tpl is None:
            tpl = TEMPLATES[self.np_random.integers(len(TEMPLATES))]
        self.instruction = options.get("instruction") or tpl.format(c=self.target)

        self._set_colours(masked=False)
        self._place_objects()
        mujoco.mj_forward(m, d)

        # start the pusher above the bench, clear of every cube
        self._tcp_target = np.array([0.34, 0.0, 0.16])
        q = self._ik(self._tcp_target)
        d.qpos[:7] = q
        d.ctrl[:7] = q
        d.ctrl[7] = 0  # fingers closed
        mujoco.mj_forward(m, d)
        for _ in range(100):  # let everything settle before the first frame
            mujoco.mj_step(m, d)

        self.t = 0
        self._settled = 0
        self._masked = False
        return self._obs(), {"target": self.target}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        tgt = self._tcp_target + a * MAX_STEP
        tgt[0] = np.clip(tgt[0], *TCP_X)
        tgt[1] = np.clip(tgt[1], *TCP_Y)
        tgt[2] = np.clip(tgt[2], TCP_Z_MIN, TCP_Z_MAX)
        self._tcp_target = tgt

        self.data.ctrl[:7] = self._ik(tgt)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        if self.mask_colour_at is not None and self.t == self.mask_colour_at:
            self._set_colours(masked=True)
            self._masked = True

        g = self.goal_xy()
        dist = float(np.linalg.norm(self.cube_xy(self.target) - g))
        on_goal = dist < GOAL_R
        self._settled = self._settled + 1 if (on_goal and self.cube_speed(self.target) < SETTLE_SPEED) else 0
        success = self._settled >= SETTLE_STEPS

        wrong = any(np.linalg.norm(self.cube_xy(n) - g) < GOAL_R
                    for n in COLOURS if n != self.target and self.distractors)
        oob = any(not (WS_X[0] - 0.12 <= self.cube_xy(n)[0] <= WS_X[1] + 0.12
                       and WS_Y[0] - 0.12 <= self.cube_xy(n)[1] <= WS_Y[1] + 0.12)
                  for n in COLOURS if (n == self.target or self.distractors))

        terminated = success
        truncated = self.t >= EPISODE_STEPS
        info = Info(self.target, success, dist, wrong, oob, self._masked).__dict__
        return self._obs(), float(success), terminated, truncated, info

    def render(self):
        return self._obs()["image"]

    def close(self):
        self._renderer.close()
