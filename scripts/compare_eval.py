import json

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
BASE = "/root/autodl-tmp/eval/baseline_{}_10ep/eval_info.json"
LORA = "/root/autodl-tmp/eval/lora3000_{}_10ep/eval_info.json"

# task names aligned with per_task_table.py (task_id order)
names10 = [
    "soup+tomato in basket", "cream cheese+butter in basket", "stove on + moka pot",
    "bowl in bottom drawer + close", "mug left plate + mug right plate", "book in back caddy",
    "mug on plate + pudding right", "soup+cream cheese in basket", "both moka pots on stove",
    "mug in microwave + close",
]

print("=== per-task comparison (baseline -> LoRA-3000, 10 episodes each) ===")
for suite in SUITES:
    b = json.load(open(BASE.format(suite)))
    l = json.load(open(LORA.format(suite)))
    bt = {t["task_id"]: t["metrics"]["successes"] for t in b["per_task"]}
    lt = {t["task_id"]: t["metrics"]["successes"] for t in l["per_task"]}
    print(f"\n--- {suite} ---")
    for tid in sorted(bt):
        bs = 100 * sum(bt[tid]) / len(bt[tid])
        ls = 100 * sum(lt[tid]) / len(lt[tid])
        d = ls - bs
        mark = " ↑" if d >= 20 else (" ↓" if d <= -20 else "")
        nm = names10[tid] if suite == "libero_10" and tid < len(names10) else f"task{tid}"
        print(f"  t{tid}: {bs:3.0f}% -> {ls:3.0f}%  ({d:+.0f}){mark}  {nm}")
