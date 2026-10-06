import json

from lerobot.datasets import LeRobotDataset

ROOT = "/root/autodl-tmp/libero_data/HuggingFaceVLA/libero"
REPO = "HuggingFaceVLA/libero"

sel = json.load(open("/root/hard_negative_episodes.json"))["episodes"]
print(f"loading {len(sel)} episodes (map-style)...", flush=True)
ds = LeRobotDataset(REPO, root=ROOT, episodes=sel)
print(f"OK: episodes={ds.num_episodes} frames={ds.num_frames}", flush=True)

s = ds[0]
shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items() if k != "task"}
print("sample[0]:", shapes, flush=True)
print("task:", s.get("task"), flush=True)

s2 = ds[len(ds) // 2]
print("mid sample OK, episode_index:", int(s2["episode_index"]), flush=True)
print("DONE_VERIFY", flush=True)
