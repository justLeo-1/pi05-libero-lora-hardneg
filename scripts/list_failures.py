import json
import os

# the six 70%-success weak tasks: (suite, task_id)
WEAK = [
    ("libero_spatial", 4), ("libero_spatial", 8),
    ("libero_goal", 6),
    ("libero_10", 4), ("libero_10", 7), ("libero_10", 8),
]

for suite, tid in WEAK:
    d = json.load(open(f"/root/autodl-tmp/eval/baseline_{suite}_10ep/eval_info.json"))
    for t in d["per_task"]:
        if t["task_id"] != tid:
            continue
        m = t["metrics"]
        for k, ok in enumerate(m["successes"]):
            if not ok:
                p = m["video_paths"][k]
                size = os.path.getsize(p) / 1e6 if os.path.exists(p) else -1
                print(f"{suite} task{tid} ep{k} FAILED size={size:.1f}MB {p}")
