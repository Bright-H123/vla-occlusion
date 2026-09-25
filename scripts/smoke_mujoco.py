"""Day 1 smoke test.

Verifies that MuJoCo can (a) load a Menagerie robot model and (b) produce an
offscreen render on this machine. Offscreen rendering is the single most likely
thing to fail on a Windows host, so it is checked before any other work.

Run:  python scripts/smoke_mujoco.py
Writes: docs/day1_render.png, docs/day1_stack.md
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = REPO / "mujoco_menagerie"
MODEL_REL = Path("franka_emika_panda") / "scene.xml"
OUT_PNG = REPO / "docs" / "day1_render.png"
OUT_MD = REPO / "docs" / "day1_stack.md"

RES = 128


def ensure_menagerie() -> Path:
    """Shallow-clone mujoco_menagerie if it is not already present."""
    model = MENAGERIE / MODEL_REL
    if model.exists():
        return model
    print("mujoco_menagerie not found, cloning (shallow) ...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/google-deepmind/mujoco_menagerie.git", str(MENAGERIE)],
        check=True,
    )
    if not model.exists():
        raise FileNotFoundError(f"cloned, but {model} is still missing")
    return model


def try_backends(model_path: Path):
    """Try each MuJoCo GL backend in turn, return (backend, rgb, timings)."""
    import mujoco  # imported here so a missing install fails loudly, after the banner

    errors = {}
    for backend in ("glfw", "egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            t0 = time.perf_counter()
            model = mujoco.MjModel.from_xml_path(str(model_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            t_load = time.perf_counter() - t0

            with mujoco.Renderer(model, height=RES, width=RES) as renderer:
                renderer.update_scene(data)
                rgb = renderer.render()
                t0 = time.perf_counter()
                for _ in range(20):
                    renderer.update_scene(data)
                    renderer.render()
                t_render = (time.perf_counter() - t0) / 20

            return backend, rgb, {
                "load_s": t_load,
                "render_s": t_render,
                "nq": model.nq,
                "nu": model.nu,
                "nbody": model.nbody,
            }
        except Exception as exc:  # noqa: BLE001 - we want the reason for every backend
            errors[backend] = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        "no MuJoCo GL backend produced an offscreen render:\n"
        + "\n".join(f"  {k}: {v}" for k, v in errors.items())
    )


def step_rate(model_path: Path, seconds: float = 2.0) -> float:
    """Physics steps per second with no rendering — the data-collection budget."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    n, t_end = 0, time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        mujoco.mj_step(model, data)
        n += 1
    return n / seconds


def main() -> int:
    print("=" * 62)
    print("Day 1 smoke test — MuJoCo offscreen rendering")
    print("=" * 62)

    model_path = ensure_menagerie()
    print(f"model: {model_path.relative_to(REPO)}")

    backend, rgb, info = try_backends(model_path)
    print(f"backend that worked: MUJOCO_GL={backend}")
    print(f"model load: {info['load_s'] * 1000:.0f} ms   "
          f"render {RES}x{RES}: {info['render_s'] * 1000:.1f} ms/frame")

    import imageio.v3 as iio
    import mujoco

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT_PNG, rgb)
    print(f"wrote {OUT_PNG.relative_to(REPO)}  ({rgb.shape[1]}x{rgb.shape[0]})")

    sps = step_rate(model_path)
    print(f"physics: {sps:,.0f} steps/s (no rendering)")

    OUT_MD.write_text(
        "\n".join([
            "# Day 1 — verified stack",
            "",
            f"Recorded by `scripts/smoke_mujoco.py` on {platform.node()}.",
            "",
            "| item | value |",
            "|---|---|",
            f"| OS | {platform.platform()} |",
            f"| Python | {sys.version.split()[0]} |",
            f"| MuJoCo | {mujoco.__version__} |",
            f"| GL backend | `MUJOCO_GL={backend}` |",
            f"| model | `{MODEL_REL.as_posix()}` (nq={info['nq']}, nu={info['nu']}, "
            f"nbody={info['nbody']}) |",
            f"| render {RES}x{RES} | {info['render_s'] * 1000:.1f} ms/frame |",
            f"| physics | {sps:,.0f} steps/s, no rendering |",
            "",
            "Offscreen rendering is checked before anything else because it is the",
            "most likely failure on a Windows host, and every later stage depends on it.",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    print("\nDay 1 local half: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
