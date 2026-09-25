#!/usr/bin/env bash
# Day 1 cloud smoke test.
#
# Run this on a rented 24 GB Linux GPU instance (L4 / A10G / 4090) BEFORE any
# other work. It answers the only question that decides the compute budget for
# the week: does lerobot train here, and what does SmolVLA actually cost per step
# on this card?
#
# Everything it prints is recorded in docs/day1_cloud.md and committed. No number
# from a vendor table or a hardware guide goes into this project; only numbers
# measured here.
#
# Usage:  bash scripts/smoke_cloud.sh 2>&1 | tee docs/day1_cloud.log

set -euo pipefail

STEPS="${STEPS:-200}"
BATCH="${BATCH:-8}"
OUT="${OUT:-/tmp/lerobot_smoke}"

banner() { printf '\n%s\n%s\n' "============================================================" "$1"; }

banner "0. machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import sys; print('python', sys.version.split()[0])"

banner "1. install lerobot"
# Pin nothing yet: record whatever resolves, then pin it in the README once it works.
pip install -q --upgrade pip
pip install -q "lerobot[smolvla]" || {
  echo "pip install failed — fall back to the official Docker image and re-run."
  exit 1
}
python - <<'PY'
import importlib.metadata as md
import torch
print("lerobot", md.version("lerobot"))
print("torch  ", torch.__version__, "cuda", torch.version.cuda)
print("device ", torch.cuda.get_device_name(0))
PY

banner "2. ACT on lerobot/pusht — does training run at all?"
/usr/bin/time -v lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht \
  --steps="${STEPS}" \
  --batch_size="${BATCH}" \
  --output_dir="${OUT}/act" \
  --wandb.enable=false 2>&1 | tail -30

banner "3. SmolVLA on lerobot/pusht — the number that sets the budget"
# n_action_steps is forced to 10. The default chunk_size of 50 is 2.5 s of
# open-loop action at 20 fps, which silently ruins any contact task.
/usr/bin/time -v lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.n_action_steps=10 \
  --dataset.repo_id=lerobot/pusht \
  --steps="${STEPS}" \
  --batch_size="${BATCH}" \
  --output_dir="${OUT}/smolvla" \
  --wandb.enable=false 2>&1 | tail -40

banner "4. peak VRAM"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

cat <<'NOTE'

============================================================
Record in docs/day1_cloud.md:
  - lerobot version that actually installed
  - seconds per step for ACT and for SmolVLA at this batch size
  - peak VRAM for each
  - extrapolated wall-clock for 20k steps of SmolVLA

If SmolVLA OOMs, degrade in this order and record which step was needed:
  batch 8 -> 4 -> 2
  resize_imgs_with_padding 512 -> 256
  steps 20k -> 10k
  lerobot[peft] LoRA (r=64, alpha=64, lr 1e-3)

STOP the instance afterwards. Do not terminate it — the environment is worth
more than the hourly charge.
============================================================
NOTE
