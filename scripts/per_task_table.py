import json

task_names = {
    "libero_spatial": [
        "bowl between plate and ramekin", "bowl next to ramekin", "bowl from table center",
        "bowl on cookie box", "bowl in top drawer", "bowl on ramekin", "bowl next to cookie box",
        "bowl on stove", "bowl next to plate", "bowl on wooden cabinet",
    ],
    "libero_object": [
        "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
        "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice",
    ],
    "libero_goal": [
        "open middle drawer", "bowl on stove", "wine on cabinet", "top drawer + bowl inside",
        "bowl on cabinet", "push plate to stove", "cream cheese in bowl", "turn on stove",
        "bowl on plate", "wine on rack",
    ],
    "libero_10": [
        "soup+tomato in basket", "cream cheese+butter in basket", "stove on + moka pot",
        "bowl in bottom drawer + close", "mug left plate + mug right plate", "book in back caddy",
        "mug on plate + pudding right", "soup+cream cheese in basket", "both moka pots on stove",
        "mug in microwave + close",
    ],
}

total_sr, total_n = 0.0, 0
for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    d = json.load(open(f"/root/autodl-tmp/eval/baseline_{s}_10ep/eval_info.json"))
    print(f"===== {s} =====")
    sr_sum, n_sum = 0.0, 0
    for t in d["per_task"]:
        succ = t["metrics"]["successes"]
        sr = 100.0 * sum(succ) / len(succ)
        name = task_names[s][t["task_id"]] if t["task_id"] < len(task_names[s]) else "?"
        flag = "  <-- failures concentrated" if sr < 80 else ""
        print(f"  task {t['task_id']}: {sum(succ)}/{len(succ)}  ({sr:.0f}%)  {name}{flag}")
        sr_sum += sr
        n_sum += 1
    print(f"  >> suite avg: {sr_sum / n_sum:.1f}%")
    total_sr += sr_sum
    total_n += n_sum
print(f"\n===== 4-suite overall average: {total_sr / total_n:.1f}% =====")
