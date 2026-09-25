# Day 1 — verified stack

Recorded by `scripts/smoke_mujoco.py` on LAPTOP-QDEMEVFL.

| item | value |
|---|---|
| OS | Windows-11-10.0.26100-SP0 |
| Python | 3.12.14 |
| MuJoCo | 3.14.0 |
| GL backend | `MUJOCO_GL=glfw` |
| model | `franka_emika_panda/scene.xml` (nq=9, nu=8, nbody=12) |
| render 128x128 | 1.2 ms/frame |
| physics | 56,048 steps/s, no rendering |

Offscreen rendering is checked before anything else because it is the
most likely failure on a Windows host, and every later stage depends on it.
