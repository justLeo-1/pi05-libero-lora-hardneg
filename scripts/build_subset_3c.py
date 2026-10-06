"""3c v2: build hard-negative episode selection, metadata-only first, then filtered load.

Avoids materializing the full parquet->arrow cache (which ENOSPC'd the system disk).
"""

import json
import random
from collections import defaultdict

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

ROOT = "/root/autodl-tmp/libero_data/HuggingFaceVLA/libero"
REPO = "HuggingFaceVLA/libero"
SEED = 42
NORMAL_PER_TASK = 8

# ---- Phase 1: metadata only (meta/*.parquet, small) ----
meta = LeRobotDatasetMetadata(REPO, root=ROOT)
print(f"meta OK: episodes={meta.total_episodes} frames={meta.total_frames} fps={meta.fps} tasks={meta.total_tasks}")

tasks_df = meta.tasks
task_names = list(tasks_df.index)
print(f"tasks table columns={list(tasks_df.columns)}")

eps = meta.episodes
print(f"episodes table: {len(eps)} rows, columns={eps.column_names}")
tname_to_idx = {}
for tname in task_names:
    tname_to_idx[tname] = int(tasks_df.loc[tname, "task_index"])

by_task = defaultdict(list)
ep_indices_col = eps["episode_index"]
tasks_col = eps["tasks"]
for i in range(len(eps)):
    t = tasks_col[i]
    tlist = t if isinstance(t, (list, tuple)) else [t]  # 'tasks' holds task name(s)
    for tn in tlist:
        by_task[tname_to_idx.get(tn, tn)].append(ep_indices_col[i])

print("\n--- all tasks ---")
for tname in task_names:
    tidx = int(tasks_df.loc[tname, "task_index"]) if "task_index" in tasks_df.columns else task_names.index(tname)
    print(f"{tidx:3d}: {len(by_task[tidx]):3d} eps  {tname}")


LIBERO10_NAMES = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove and put the moka pot on it",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both moka pots on the stove",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "pick up the book and place it in the back compartment of the caddy",
}


def norm(s):
    return s.lower().replace(" ", "_").replace("-", "_").strip()


def is_hard(tname):
    t = norm(tname)
    return (
        tname.strip().lower() in LIBERO10_NAMES  # libero_10 (dataset indices 0-9)
        or "top_drawer_of_the_wooden_cabinet" in t  # spatial t4
        or t == "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"  # spatial t8
        or t == "put_the_cream_cheese_in_the_bowl"  # goal t6
    )


rng = random.Random(SEED)
hard_tasks, normal_tasks = [], []
hard_eps, normal_eps = [], []
for tname in task_names:
    tidx = int(tasks_df.loc[tname, "task_index"]) if "task_index" in tasks_df.columns else task_names.index(tname)
    eps_list = sorted(by_task[tidx])
    if is_hard(tname):
        hard_tasks.append((tidx, tname, len(eps_list)))
        hard_eps.extend(eps_list)
    else:
        normal_tasks.append((tidx, tname, len(eps_list)))
        normal_eps.extend(sorted(rng.sample(eps_list, min(NORMAL_PER_TASK, len(eps_list)))))

sel = sorted(set(hard_eps) | set(normal_eps))
print(f"\nSELECTION: hard_tasks={len(hard_tasks)} hard_eps={len(hard_eps)} | "
      f"normal_tasks={len(normal_tasks)} normal_eps={len(normal_eps)} | total={len(sel)} "
      f"| normal_ratio={len(normal_eps) / len(sel):.1%}")

out = {
    "seed": SEED,
    "normal_per_task": NORMAL_PER_TASK,
    "hard_tasks": [{"task_index": i, "name": n, "n_eps": c} for i, n, c in hard_tasks],
    "n_hard": len(hard_eps),
    "n_normal": len(normal_eps),
    "episodes": sel,
}
with open("/root/hard_negative_episodes.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved /root/hard_negative_episodes.json")

# ---- Phase 2: streaming sanity check (no arrow materialization) ----
from lerobot.datasets import StreamingLeRobotDataset

ds2 = StreamingLeRobotDataset(REPO, root=ROOT, episodes=sel)
print(f"streaming dataset ready (meta reports full set: {ds2.num_episodes} eps)")
sel_set = set(sel)
seen_eps, n_ok = set(), 0
first_sample = None
for s in ds2:
    if first_sample is None:
        first_sample = s
    e = int(s["episode_index"].item()) if hasattr(s["episode_index"], "item") else int(s["episode_index"])
    assert e in sel_set, f"UNFILTERED episode {e} appeared!"
    seen_eps.add(e)
    n_ok += 1
    if n_ok >= 100:
        break
shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in first_sample.items() if k != "task"}
print(f"first {n_ok} samples OK, {len(seen_eps)} distinct episodes, all within selection")
print("sample[0]:", shapes)
print("sample task:", first_sample.get("task"))
print("DONE_3C")
