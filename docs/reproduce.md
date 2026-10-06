# Reproduction Guide

## 1. Environment

- Linux + CUDA GPU (≥16 GB VRAM for evaluation; training measured at only 8.9 GB)
- conda env: Python 3.12 + torch 2.11 (tested with cu130)
- [LeRobot](https://github.com/huggingface/lerobot) editable install from source (tested at commit `6adf515`) + `lerobot[dataset]` + `transformers==5.5.4`
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): the official `setup.py` is broken (the outer `libero/` has no `__init__.py`) — do **not** `pip install -e`; instead drop a `libero_dev.pth` pointing at the repo root into site-packages
- Dependencies: `mujoco==3.2.3` + `robosuite==1.4.1` + `gym==0.26.2` + `bddl` + `easydict` + `matplotlib`, plus the system package `libegl1` (EGL offscreen rendering)
- paligemma tokenizer: `google/paligemma-3b-pt-224` is a gated model (mirrors may return 403); alternatives: ModelScope `AI-ModelScope/paligemma-3b-pt-224`, or the public GCS bucket `https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`

## 2. Data

```bash
# dataset (33 GB)
huggingface-cli download HuggingFaceVLA/libero --repo-type dataset --local-dir <DATA_ROOT>

# build the hard-negative selection (metadata-only; does not materialize image frames)
python scripts/build_subset_3c.py   # edit ROOT in the script to point at DATA_ROOT
# output: hard_negative_episodes.json (already shipped in data/, so this step is optional)

# verify filtered-subset loading
python scripts/verify_subset_load.py
```

Note: image frames are embedded inside parquet rows in this dataset; a full map-style load materializes an ~18 GB arrow cache — reserve disk space or point `HF_DATASETS_CACHE` at a large volume.

## 3. Training

```bash
lerobot-train \
  --job_name=lora_hardneg_6k \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.dtype=bfloat16 \
  --dataset.root=<DATA_ROOT> \
  --dataset.episodes="$(cat data/hard_negative_episodes.json)" \
  --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=64 \
  --batch_size=8 --steps=6000 --save_freq=1500 --log_freq=100 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/lora_hardneg_6k
```

The full configuration (optimizer/scheduler/LR etc.) is authoritative in `configs/train_config.json`; training curves in `results/curves/`.

Checkpoint selection: the loss plateaus from ~step 1700 (narrow 0.030–0.036 oscillation), so the step-3000 checkpoint was evaluated. The 1500/4500/6000-step checkpoints were kept but not evaluated.

## 4. Evaluation

Baseline:

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.dtype=bfloat16 \
  --env.type=libero --env.task=libero_10 \
  --eval.n_episodes=10 --eval.batch_size=8 --eval.use_async_envs=true \
  --output_dir=outputs/eval/baseline_libero_10_10ep
```

LoRA-3000 (same, only the policy path changes):

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=checkpoints/lora3000 \
  --policy.dtype=bfloat16 \
  --env.type=libero --env.task=libero_10 \
  --eval.n_episodes=10 --eval.batch_size=8 --eval.use_async_envs=true \
  --output_dir=outputs/eval/lora3000_libero_10_10ep
```

- `--env.task` options: `libero_spatial` / `libero_object` / `libero_goal` / `libero_10`; add `--env.task_ids='[0]'` for a single task
- **`--eval.n_episodes` is per task**: a full suite with n=10 is 100 episodes
- Always run long jobs under `nohup` in the background
- Full protocol: `docs/eval_protocol.md`

## 5. Result analysis

```bash
python scripts/list_failures.py    # list failure episodes from eval_info.json
python scripts/per_task_table.py   # per-task success-rate table
python scripts/compare_eval.py     # baseline vs LoRA comparison
```
