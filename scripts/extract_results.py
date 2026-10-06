import ast
import re

for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    print(f"===== {s} =====")
    txt = open(f"/root/eval_baseline_{s}.log", errors="replace").read().replace("\r", "\n")
    seen = set()
    for line in txt.split("\n"):
        if "bot_eval.py" not in line or "{" not in line:
            continue
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, dict) and "pc_success" in v:
                key = (k, v.get("n_episodes"))
                if key in seen:
                    continue
                seen.add(key)
                print(f"{k}: pc_success={v['pc_success']} reward={v.get('avg_sum_reward')} n={v.get('n_episodes')}")
