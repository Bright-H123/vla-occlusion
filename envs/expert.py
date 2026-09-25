"""Scripted pushing expert with privileged state.

Closed-loop state machine, re-aimed every control step from the true cube and
goal positions:

    LIFT     raise the pusher clear of every cube before moving sideways
    TRAVEL   move at height to a staging point behind the target cube
    DESCEND  lower to pushing height at the staging point
    PUSH     drive the cube along the cube-to-goal line, correcting laterally
    DONE     back off and hold so the cube can settle on the goal

If the pusher loses the push line while pushing (the cube slid or rotated), the
expert goes back to LIFT and re-stages rather than dragging the cube sideways.

The expert reads privileged state. The policies trained on its demonstrations see
only the camera image, the TCP state and the instruction.
"""

from __future__ import annotations

import numpy as np

from envs.select_push import CUBE_HALF, GOAL_R, MAX_STEP, PUSHER_R

HIGH_Z = 0.10          # pusher tip height when travelling over cubes
LOW_Z = 0.012          # pushing height, below the cube's centre of mass
CONTACT = CUBE_HALF + PUSHER_R           # centre-to-centre at contact
STAGE = CONTACT + 0.030                  # staging distance behind the cube
PRESS = 0.008                            # how far inside the contact point to aim
ALIGN_TOL = 0.016                        # lateral error that triggers a re-stage
POS_TOL = 0.006
PUSH_SPEED = 0.55                        # fraction of MAX_STEP while pushing
DONE_DIST = GOAL_R * 0.45


class PushExpert:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.phase = "LIFT"
        self.restages = 0

    def _toward(self, cur: np.ndarray, tgt: np.ndarray, speed: float = 1.0) -> np.ndarray:
        step = (tgt - cur) / MAX_STEP
        n = np.linalg.norm(step)
        if n > speed:
            step = step / n * speed
        return step

    def act(self) -> np.ndarray:
        env = self.env
        sp = env._tcp_target            # commanded setpoint; the action moves this
        c = env.cube_xy(env.target)
        g = env.goal_xy()

        to_goal = g - c
        dist = float(np.linalg.norm(to_goal))
        u = to_goal / max(dist, 1e-6)
        n = np.array([-u[1], u[0]])      # unit normal to the push line

        stage_xy = c - u * STAGE
        rel = sp[:2] - c
        along = float(rel @ u)
        lateral = float(rel @ n)

        if dist < DONE_DIST:
            self.phase = "DONE"

        if self.phase == "DONE":
            back = np.array([*(c - u * (CONTACT + 0.03)), LOW_Z + 0.03])
            return np.clip(self._toward(sp, back, 0.4), -1, 1).astype(np.float32)

        if self.phase == "PUSH":
            lost = abs(lateral) > ALIGN_TOL or along > -CONTACT + 0.012
            if lost:
                self.phase = "LIFT"
                self.restages += 1
            else:
                # aim a little inside the back face, pushed along u, with lateral correction
                aim = c - u * (CONTACT - PRESS) - n * lateral * 0.6
                aim = aim + u * MAX_STEP * PUSH_SPEED
                slow = min(1.0, dist / 0.05)          # ease off near the goal
                tgt = np.array([*aim, LOW_Z])
                return np.clip(self._toward(sp, tgt, PUSH_SPEED * max(slow, 0.35)),
                               -1, 1).astype(np.float32)

        if self.phase == "LIFT":
            if sp[2] >= HIGH_Z - POS_TOL:
                self.phase = "TRAVEL"
            else:
                return np.clip(self._toward(sp, np.array([sp[0], sp[1], HIGH_Z])),
                               -1, 1).astype(np.float32)

        if self.phase == "TRAVEL":
            if np.linalg.norm(sp[:2] - stage_xy) < POS_TOL:
                self.phase = "DESCEND"
            else:
                return np.clip(self._toward(sp, np.array([*stage_xy, HIGH_Z])),
                               -1, 1).astype(np.float32)

        if self.phase == "DESCEND":
            if sp[2] <= LOW_Z + POS_TOL:
                self.phase = "PUSH"
            else:
                return np.clip(self._toward(sp, np.array([*stage_xy, LOW_Z])),
                               -1, 1).astype(np.float32)

        return np.zeros(3, np.float32)
