# Language-Conditioned Selective Manipulation Under Occlusion

A custom MuJoCo scene in which a robot must push **the object named in a
natural-language instruction** to a goal region, while visually similar
distractors are present and the target is temporarily hidden partway through
the episode. Instantiated as a greenhouse bench, but the question is not
agricultural: it is whether a fine-tuned vision-language-action policy actually
uses the instruction, and whether it still does once the target leaves view.

> **Status.** The scene, the scripted expert, the null controls, the occluder, the
> dataset recorder and the four-condition evaluation harness are built and
> measured; everything claimed about them below is a number this repository
> produced. **No policy result is reported yet.** The ACT baseline has only been
> trained to 10k steps, about 5.4 epochs, and at that point it reaches 1/10 on the
> matched condition. A policy that cannot do the task cannot be asked whether it
> uses the instruction, so the policy row of the results table is deliberately
> empty rather than filled with numbers from an undertrained model.
>
> Built with AI assistance. The experimental design, the controls and every
> reported number are mine.
> Nothing in this README is reported until the code that produces it is in this
> repository.

<!-- Day 7: success GIF and failure GIF go here, above everything else. -->

## Scope

**What this does**

- A self-built MJCF scene (Gymnasium wrapper) with randomised object layout,
  randomised colour-to-position assignment, and an occluder panel that moves onto
  the line between the front camera and the bench mid-episode. The panel is visual
  only (no collision), so it removes the observation without changing the physics.
- A scripted end-effector expert that solves the task from privileged state, used
  to generate demonstrations automatically. No human teleoperation.
- Fine-tuning of the **action expert** of SmolVLA-450M on those demonstrations.
- An ACT policy trained on the same data as a **no-language baseline**.
- A four-condition evaluation with fixed trial counts, multiple seeds,
  confidence intervals and a failure taxonomy.

**What this does NOT do**

- No grasping. The task is pushing, so no gripper contact parameters are involved.
- No real robot, no sim-to-real, no domain randomisation over dynamics, no calibration.
- No mobile base, no navigation.
- The SmolVLA vision-language backbone is **not** trained. `freeze_vision_encoder=True`
  and `train_expert_only=True` are left at their defaults, so roughly 100M parameters
  of flow-matching action expert are updated and the ~350M backbone is not.
  This is fine-tuning an action expert, not training a VLA.
- No generalisation claims. Instructions come from a fixed set of templates
  (listed in full below) and objects are the colours seen in training.
- No comparison against pi0.5 / X-VLA / GR00T on hardware. Why those were excluded
  is documented in [Landscape](#landscape) with VRAM thresholds and sources.

## The question

A policy that reaches a high success rate on a language-conditioned task has not
thereby been shown to use the language. It may have learned the visual layout, or
a colour prior, or the position the target usually occupies. Four conditions
separate these:

| # | Condition | What it isolates |
|---|---|---|
| a | Matched instruction | Baseline performance |
| b | **Swapped instruction** — same initial state, instruction names a different object, success still scored against the instructed target | Whether the instruction is used at all |
| c | No distractor — only the target is present | Whether language is needed for this scene at all |
| d | **Occlusion** — the target is visible when the instruction is given, then hidden | Whether the policy retains the target across a gap in observation |

Condition (b) is the identity condition for ACT, which has no language input.
That is the point of including it.

Condition (d) is the one this project exists for. SmolVLA has no explicit memory,
so a drop here is the expected result rather than a failure, and it is the
minimal case that motivates memory-conditioned policies.

**How (d) is implemented.** `SelectPushEnv(occlude_at=t0, occlude_steps=k)` moves a
grey panel onto the front camera's line of sight for steps `t0 <= t < t0 + k`, then
parks it outside the view again. Measured on seed 7 by differencing identical
episodes with and without the panel: it changes 93.8% of the 128x128 front frame,
and the pixel count of every cube colour drops to 0 while it is in place (red 92,
green 28, blue 14 pixels before). It is built for the default `front` camera; on
the `top` camera it changes only 13% of the frame, so (d) is only defined for
`front`. `info["occluded"]` reports the state at every step.

A weaker variant, `mask_colour_at=t`, keeps the cubes visible but recolours all
three grey. It removes identity and keeps geometry, whereas the panel removes
both, so the two together separate "lost track of which cube" from "lost sight
of the scene".

## Results

### What is measured

| Claim | Measurement |
|---|---|
| The scripted expert solves the task | **100/100** episodes, seeds 1000-1099, Wilson 95% [96%, 100%] |
| Success is not reachable by chance | random actions **0/50**, zero actions **0/50**, seeds 5000-5049 |
| Success is bound to the *instructed* cube | expert pushing a distractor **0/50**, expert pushing the instructed cube **50/50** |
| The occluder really removes the observation | **93.8%** of the 128x128 front frame changes, and the pixel count of every cube colour falls to 0 (red 92, green 28, blue 14 before) |
| The occluder is front-camera only | it changes **13%** of the top-camera frame, so condition (d) is defined for `front` alone |
| A policy checkpoint carries usable normalisation statistics | `scripts/check_norm_stats.py`, confirmed to pass a fine-tuned checkpoint, to fail the released `lerobot/smolvla_base` (which ships only `so100.buffer.action.*`), and to fail a checkpoint against the wrong dataset |
| Paired conditions are equally feasible | evaluation layouts require **all three** push lanes clear, not just the target's, which 56% of seeds satisfy |

The demonstration set is 180 episodes, 180/180 successful, 14905 frames, 12 MB,
balanced 61 red / 60 green / 59 blue over 9 distinct instructions.

### What is not measured yet

| Policy | (a) matched | (b) swapped | (c) no distractor | (d) occluded |
|---|---|---|---|---|
| ACT (no language), 10k steps | 1/10 | not run | not run | not run |
| ACT, trained to convergence | not run | not run | not run | not run |
| SmolVLA (action expert fine-tuned) | not run | not run | not run | not run |

The 10k-step ACT number is there to say why the rest is blank, not as a result
about ACT. Nine of its ten failures were timeouts.

## Instruction templates

All three templates, verbatim from `envs/select_push.py`:

```
push the {c} cube to the goal
move the {c} block onto the target
slide the {c} cube onto the pad
```

`{c}` is one of `red`, `green`, `blue`, giving nine distinct instructions. There
are **no unseen phrasings and no unseen colours** in the evaluation. Nothing here
tests language generalisation, and no such claim is made.

## Reproducing

```bash
# verify the stack and the expert before anything is trained on it
python scripts/smoke_mujoco.py
python scripts/run_expert.py --n 100 --record     # the 100/100 gate
python scripts/null_controls.py --n 50            # the three null controls

# record demonstrations
python data/record.py --n 180 --out data/scripted_v1 --seed0 3000

# train the no-language baseline locally
python -m lerobot.scripts.lerobot_train --policy.type=act   --dataset.repo_id=local/select_push --dataset.root=data/scripted_v1   --policy.chunk_size=50 --policy.n_action_steps=10   --batch_size=8 --steps=100000 --num_workers=4   --policy.device=cuda --dataset.video_backend=pyav   --output_dir=outputs/act_scripted_v1

# check the checkpoint before trusting any evaluation from it
python scripts/check_norm_stats.py   outputs/act_scripted_v1/checkpoints/<step>/pretrained_model   --dataset-root data/scripted_v1

# four conditions on paired layouts
python scripts/evaluate_policy.py   outputs/act_scripted_v1/checkpoints/<step>/pretrained_model   --n 60 --conditions a b c d
```

`--policy.n_action_steps=10` rather than the default 50: at 20 Hz a 50-step chunk
is 2.5 s of open-loop motion, and the episodes average 83 steps, so the default
would cover most of a trial without looking at the scene again.

On Windows, `--policy.path=lerobot/smolvla_base` fails with an `HFValidationError`
because the hub id is turned into a `WindowsPath`; pass a local checkpoint
directory instead. Every checkpoint save also raises `WinError 1314` on the
`checkpoints/last` symlink, which needs a privilege Windows does not grant by
default; the checkpoint itself is written before that happens and loads fine.
`docs/smolvla_deploy_windows.md` has both, with the measured numbers.

## Hardware and budget

Measured on this project, on an RTX 4060 Laptop (8188 MiB, driver 560.94),
with torch 2.11.0+cu126, lerobot 0.6.1, mujoco 3.14.0.

| | SmolVLA 450M | ACT 51.6M |
|---|---|---|
| checkpoint load | 7.8 s | 0.8 s |
| one 50-action chunk | 244 ms | 8 ms |
| serving a queued action | 2.4 ms | 0.8 ms |
| inference peak VRAM | 916 MiB | 216 MiB |
| fine-tune at batch 8 | 2.33 GB, 4.4 steps/s | 0.94 GB, 5.1 steps/s |
| fine-tune at batch 32 | 5.30 GB, 1.05 steps/s | 1.44 GB, 2.1 steps/s |

Both fit the 8 GB card, so no cloud GPU was needed for either. Past about 7 GB
this card does not raise `torch.OutOfMemoryError`: at batch 48 SmolVLA reached
7.31 GB and the step time went from 0.67 s to 2.3-4.4 s while the run still
exited 0, so every local training run sets
`torch.cuda.set_per_process_memory_fraction` to make the overflow fail loudly.

The policy can only re-plan about 3.5 times a second, so the achievable
closed-loop rate is 3.5 Hz times `n_action_steps`. At 10 that is a 0.33 s
open-loop window, which is why 10 was chosen over the default 50.

Cloud cost so far: nothing.

## Landscape

Why SmolVLA and not a larger model, with the threshold that rules each one out
on the hardware actually available here:

| Model | Inference | Fine-tune | Verdict |
|---|---|---|---|
| **SmolVLA 450M** | about 2 GB | 2.3 GB at batch 8, measured | the only one that both infers and fine-tunes on an 8 GB laptop |
| pi0 / pi0.5 | 14 GB and up | LoRA over 22.5 GB, full over 70 GB; the published pi0.5 recipe is sized for 80 GB | inference alone exceeds this card |
| GR00T N1.5 | 16 GB and up | about 25 GB, needs `--no-tune_diffusion_model` on a 4090, LoRA wants two | out of reach |

Sources are the openpi repository's own memory table, the LeRobot hardware
guide, and the GR00T fine-tuning notes. The SmolVLA figures in the table above
are measured here, not quoted.

One thing the released checkpoint does not provide: `lerobot/smolvla_base` ships
no usable normalisation statistics. Its normaliser holds only
`so100.buffer.action.mean/std`, `so100-red.buffer.*` and `so100-blue.buffer.*`,
with no plain `observation.state.*` or `action.*` key at all. A missing statistic
does not raise, it is silently skipped, so the policy trains and runs on
unnormalised state and simply performs badly. That is what
`scripts/check_norm_stats.py` exists to catch.

## Attribution

- Robot model from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (Google DeepMind). Scene, task, expert and evaluation code in this repository are
  written for this project.
- Policies and training entry points from [LeRobot](https://github.com/huggingface/lerobot)
  (Hugging Face, Apache-2.0); `lerobot/smolvla_base` weights Apache-2.0.
- Licences of all third-party components are preserved in `THIRD_PARTY.md`.
