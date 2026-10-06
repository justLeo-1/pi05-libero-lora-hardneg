# Evaluation Protocol (identical for Baseline and LoRA-3000)

Cite this protocol alongside any success-rate number from this repository.

- **Hardware**: single RTX 4090 24GB
- **Precision**: bfloat16 (the checkpoint's native storage format; ~2× speedup from 4090 tensor cores)
- **Parallelism**: `batch_size=8`, `use_async_envs=true` (async stepping; the synchronous mode is bottlenecked by CPU-side rendering — async is 15–20× faster)
- **Scale**: 4 suites × 10 tasks × 10 episodes = **400 episodes**; 280-step cap per episode
- **Seed**: not fixed (`start_seed=None`, initial states sampled randomly)
- **Policy inference parameters** (intrinsic to the checkpoint's config.json): flow-matching with 10 Euler integration steps; action chunk = 50, fully executed (one inference per 50 env steps); dual cameras 256×256→224×224 plus one empty-camera placeholder; mean/variance normalization on actions and states; `compile_model=true (max-autotune)`

## Statistical caveats

- Per-task 10 episodes → 95% CI roughly ±15–30pp (coarse)
- Suite-level 100 episodes → ~±6pp
- Recommended scale-up: 20 episodes/task on the suites of interest (the LIBERO standard protocol)

## Runtime reference

| Suite | Baseline | LoRA-3000 |
|---|---|---|
| libero_spatial | 12.8 min | — |
| libero_object | 13.2 min | — |
| libero_goal | 11.7 min | — |
| libero_10 | 17.7 min | — |
| Total | ~55 min | ~74 min |

~5.6 s per episode amortized (baseline).
