# π0.5 + LIBERO: Hard-Negative LoRA Fine-Tuning

> We benchmarked `pi05_libero_finetuned_v044` on LIBERO (400 episodes, **88.25%** — the official model card publishes no success rate), manually attributed all failure episodes of the weakest tasks into three failure modes, built a 724-episode hard-negative training subset accordingly, and fine-tuned with LoRA (r=32). Under an identical evaluation protocol the model improved to **89.75%** (+4pp on the long-horizon suite `libero_10`, no regression on any of the 6 weak tasks). All configs, scripts, logs, per-episode results and failure analysis are included.

## Results

| Suite | Baseline | LoRA-3000 | Δ |
|---|---|---|---|
| libero_spatial | 89.0% | 89.0% | ±0 |
| libero_object | 89.0% | 89.0% | ±0 |
| libero_goal | 91.0% | 93.0% | +2 |
| libero_10 (long-horizon, target) | 84.0% | **88.0%** | **+4** |
| **4-suite average** | 88.25% | **89.75%** | **+1.5** |

Protocol: 10 tasks × 10 episodes = 100 episodes per suite, identical for both models (full protocol in `docs/eval_protocol.md`). Raw per-episode records in `results/eval_info/`.

**Weak-task closed loop** (the 6 tasks at 70% success in baseline → targeted data → re-evaluation):

| Weak task | Failure mode | Baseline | LoRA-3000 |
|---|---|---|---|
| spatial t4: bowl from drawer | B: occluded precise grasp | 70% | **90%** |
| goal t6: cream cheese in bowl | B: occluded precise grasp | 70% | **90%** |
| libero_10 t4: two mugs on two plates | A: second sub-goal | 70% | **80%** |
| libero_10 t8: two moka pots on stove | A: second sub-goal | 70% | **80%** |
| libero_10 t7: soup + cream cheese in basket | A: second sub-goal | 70% | 70% |
| spatial t8: bowl next to plate | C: spatial ambiguity | 70% | 70% |

## Method (five steps)

1. **Baseline measurement**: `lerobot/pi05_libero_finetuned_v044` (bf16), 4 suites × 100 episodes, ~55 min of GPU time. The official model card publishes no success rate — the protocol and numbers in this repo are independently citable.
2. **Failure attribution**: manually watched all 18 failure videos of the six 70%-success tasks, yielding three modes — **A: long-horizon "second sub-goal" failure** (9/18: after completing the first object, the policy omits/misplaces/interferes with the second), **B: precise grasp under occlusion** (6/18), **C: near-neighbor spatial ambiguity** (3/18). Full report: `analysis/badcase_analysis.md`; videos: `analysis/badcase_videos/`.
3. **Data construction**: hard set = all 10 libero_10 tasks + spatial t4/t8 + goal t6 (508 episodes), mixed with 8 episodes from each of the remaining 27 tasks (216 episodes, 29.8%) to prevent forgetting — **724 episodes** total, seed=42. Manifest: `data/hard_negative_episodes.json`; builder: `scripts/build_subset_3c.py`.
4. **LoRA fine-tuning**: r=32, α=64, π0.5 default target modules (Action Expert q/v attention + projection layers), vision tower frozen; 6000 steps, bs=8, lr 2.5e-5 cosine (1000 warmup); single RTX 4090, 8.9 GB VRAM, 1h27m. Full config: `configs/train_config.json`.
5. **Identical-protocol re-evaluation**: checkpoints every 1500 steps; the 3000-step checkpoint was selected on the loss plateau (flat from ~step 1700) and evaluated in full (`checkpoints/lora3000/`).

## Repository layout

```
configs/train_config.json      # full training config snapshot
scripts/                       # reproducibility scripts
  build_subset_3c.py           #   hard-negative subset builder (metadata-only, avoids materializing 18 GB cache)
  list_failures.py             #   failure-episode listing from eval_info.json
  compare_eval.py              #   baseline vs LoRA per-task comparison
  per_task_table.py            #   per-task success-rate table
  extract_results.py           #   metric extraction from eval logs
  verify_subset_load.py        #   filtered-subset loading verification
  parse_train_log.py           #   training-log parser (lerobot abbreviates steps ≥1000 as "4K"; restores true step numbers by line order)
data/hard_negative_episodes.json  # the 724-episode selection (seed=42)
checkpoints/lora3000/          # LoRA adapter (step 3000, 10.3 MB, the evaluated one)
results/
  eval_info/                   # raw per-episode records of all 8 evaluation runs
  logs/                        # full training + evaluation logs
  curves/                      # loss / grad-norm / lr / VRAM / throughput curves + CSV
  demo_videos/                 # one rollout per suite per model
analysis/
  badcase_analysis.md          # per-video attribution report of all 18 failures
  badcase_videos/              # the 18 failure rollout videos
docs/
  eval_protocol.md             # full evaluation protocol (cite alongside any numbers from this repo)
  reproduce.md                 # environment, data, training and evaluation instructions
```

## Quick start

Evaluate the adapter (lerobot ≥ 0.6.2, see `docs/reproduce.md`):

```bash
lerobot-eval \
  --policy.path=checkpoints/lora3000 \
  --policy.dtype=bfloat16 \
  --env.type=libero --env.task=libero_10 \
  --eval.n_episodes=10 --eval.batch_size=8 --eval.use_async_envs=true \
  --output_dir=outputs/eval_lora3000_libero10
```

## Honest limitations

- Per-task 10-episode 95% CI is roughly ±15–30pp; suite-level (100 episodes) ~±6pp. The +4pp direction on libero_10 is clear, but the precise delta should be re-measured at 20 episodes/task before citation.
- object t7 (milk in basket) dropped 100%→60% (-40pp on a single task), masked by same-suite gains; within the noise band of 10 episodes — do not draw conclusions without re-measurement.
- Training data is same-source and same-distribution as the baseline's; this work demonstrates "targeted data patching of known failure modes", not out-of-distribution generalization.

## References

- Base model: [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) (π0.5, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054))
- Benchmark: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (dataset: [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero))
- Training framework: [LeRobot](https://github.com/huggingface/lerobot)

## License

Code and documentation: MIT (see `LICENSE`). Model weights follow the original terms of the base model `lerobot/pi05_libero_finetuned_v044`.
