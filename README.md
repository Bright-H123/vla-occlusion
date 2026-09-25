# Language-Conditioned Selective Manipulation Under Occlusion

A custom MuJoCo scene in which a robot must push **the object named in a
natural-language instruction** to a goal region, while visually similar
distractors are present and the target is temporarily hidden partway through
the episode. Instantiated as a greenhouse bench, but the question is not
agricultural: it is whether a fine-tuned vision-language-action policy actually
uses the instruction, and whether it still does once the target leaves view.

> **Status: in progress.** Numbers below marked `TBD` have not been measured yet.
> Nothing in this README is reported until the code that produces it is in this
> repository.

<!-- Day 7: success GIF and failure GIF go here, above everything else. -->

## Scope

**What this does**

- A self-built MJCF scene (Gymnasium wrapper) with randomised object layout,
  randomised colour-to-position assignment, and an occluder that passes between
  the camera and the workspace mid-episode.
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

## Results

`TBD` — populated on Day 6 from `results/*.csv`. Every cell will carry n and a
Wilson 95% interval, and will be reproducible with the command in
[Reproducing](#reproducing).

| Policy | (a) matched | (b) swapped | (c) no distractor | (d) occluded |
|---|---|---|---|---|
| Scripted expert | TBD | — | TBD | — |
| ACT (no language) | TBD | TBD | TBD | TBD |
| SmolVLA (action expert fine-tuned) | TBD | TBD | TBD | TBD |

Failure taxonomy: `TBD`.

## Instruction templates

`TBD` — all templates will be listed here verbatim. There are no unseen phrasings
in the evaluation, and that is stated rather than implied.

## Reproducing

`TBD` — three copy-pasteable commands: generate data, train, evaluate.

## Hardware and budget

`TBD` — measured on this project, not quoted from a vendor table.

## Landscape

`TBD` — why SmolVLA and not pi0.5 / X-VLA / GR00T N1.7 / OpenVLA-OFT, with the
VRAM threshold and a primary source for each.

## Attribution

- Robot model from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (Google DeepMind). Scene, task, expert and evaluation code in this repository are
  written for this project.
- Policies and training entry points from [LeRobot](https://github.com/huggingface/lerobot)
  (Hugging Face, Apache-2.0); `lerobot/smolvla_base` weights Apache-2.0.
- Licences of all third-party components are preserved in `THIRD_PARTY.md`.
