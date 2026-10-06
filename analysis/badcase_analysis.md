# Bad-Case Attribution Analysis: pi05_libero_finetuned_v044 @ LIBERO

> Written on 2026-08-27 (the night of the baseline evaluation).
> Scope: **all 18 failure episodes** of the six weakest tasks (70% success rate) in the baseline evaluation (protocol: `../docs/eval_protocol.md`).
> Video files: `badcase_videos/` in this directory (filename = `{suite}_{task_id}_ep{k}.mp4`, one-to-one with the baseline eval output directories).

## 1. Background and sample composition

- Overall baseline: 88.2% (400 episodes, 48 failures), unevenly distributed
- This analysis covers the six 70%-success tasks × 3 failures each = **18 episodes** (37.5% of all failures):
  - libero_10 tasks 4 / 7 / 8 (long-horizon two-object tasks, 3 failures each)
  - libero_spatial tasks 4 / 8 (bowl from drawer; bowl next to plate, 3 failures each)
  - libero_goal task 6 (cream cheese into bowl, 3 failures)
- The remaining 30 failures are scattered across 80–90%-success tasks and were not analyzed one by one (can be added if needed)
- Method: manually watched every third-person-view rollout video (280-step cap, compressed low resolution) and classified by failure stage and error type

## 2. Failure-mode taxonomy (core findings)

### Mode A: long-horizon "second sub-goal" failure — 9/18, all in libero_10

Every failure of the two-object tasks happens **after the first sub-goal completes**, in three forms:

| Form | Description | Seen in |
|---|---|---|
| A1 omission | after placing the first object, never handles the second, or keeps touching the completed target until timeout | two-mug, two-can, two-pot tasks |
| A2 misplacement | the second object is placed but in the wrong spot (e.g. both mugs on one plate, left/right assignment fails) | mainly the two-mug task |
| A3 interference | while placing the second object, knocks the first one over / out of the target zone | moka pots on stove, cans in basket |

**Mechanism analysis**: the model is not "incapable of the task" — the first stage (grasp + place the first object) mostly succeeds even in failure cases. What fails is **retaining the second-half intent after completing the first half**. The policy outputs a 50-step action chunk per inference (chunk=50 @10fps ≈ 5 s), and the sub-goal switch happens right around chunk boundaries; intent retention across chunks is the weak point. This matches the π0.5 paper's discussion that long-horizon tasks rely on hierarchy/memory mechanisms (this checkpoint has only the flow-matching head, no sub-task prediction component).

### Mode B: precise-grasp failure under occlusion — 6/18

- **Bowl from drawer (spatial task 4)**: reaching into the drawer, the gripper catches the rim and slips, or drops the bowl on the way out — **precise manipulation under occlusion**
- **Cream cheese into bowl (goal task 6)**: clumsy grasping of the small object (repeated pose adjustments), placement lands outside the bowl rim

### Mode C: near-neighbor spatial ambiguity — 3/18 (spatial task 8, bowl next to plate)

Among several black bowls on the table, the policy **picks the wrong target** (grabs a bowl already on the plate, or a farther one), or drags/misaligns before grasping. This is a grounding problem for the spatial reference "next to the plate"; manipulation ability itself is fine.

## 3. Per-task breakdown

| Task | Instruction | Failure eps | Attribution |
|---|---|---|---|
| libero_10 t4 | white mug on left plate + yellow-white mug on right plate | ep1/2/4 | A (A1 omission, A2 left/right mix-up) |
| libero_10 t7 | alphabet soup + cream cheese, both in basket | ep0/6/8 | A (A1 omission, A3 knock-out) |
| libero_10 t8 | both moka pots on stove | ep2/3/8 | A (A3 knock-over, unstable grasp of 2nd pot) |
| libero_spatial t4 | bowl from top drawer of wooden cabinet onto plate | ep2/3/7 | B (occluded precise grasp slips) |
| libero_spatial t8 | bowl next to plate onto plate | ep4/5/8 | C (wrong bowl among near neighbors) |
| libero_goal t6 | cream cheese into bowl | ep0/2/4 | B (small-object grasp/place) |

## 4. Input to the hard-negative data strategy

1. **Hard task set**: demonstrations of all 10 libero_10 tasks (multi-task mixing regularizes; same long-horizon distribution) + spatial t4/t8 + goal t6 ≈ 13 tasks ≈ 550 episodes
2. **Anti-forgetting mix-in**: 150–200 episodes sampled from the other 27 tasks (~30% share)
3. **Mechanism match**: demonstrations naturally contain "successful second-sub-goal switch" samples, directly filling the Mode-A data gap; Mode-B/C tasks benefit along the way
4. **Evaluation hook**: after fine-tuning, watch per-task success on libero_10 (especially the three Mode-A tasks) for 70%→90%+, with no regression on the other suites

## 5. Limitations

- Videos are compressed, low-resolution, third-person view only (no wrist-camera view); the exact failure stage of individual episodes may be misjudged — spot-checking 2–3 raw videos is advised
- The 30 failures in 80–90% tasks were not analyzed; mode proportions only represent the 70%-task subset
- Manual classification without frame-level annotation; a stronger report could add timeline annotations (the step at which deviation begins)

## 6. Reproduction

`../scripts/list_failures.py` generates the failure list from each suite's `eval_info.json` (which contains per-episode success flags and video paths); the videos in `badcase_videos/` were copied from the corresponding baseline eval output directories.
