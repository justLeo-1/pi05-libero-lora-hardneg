"""Parse lerobot-train INFO metric lines into CSV + curve plots.

Usage: python parse_train_log.py <train.log> <out_dir>
Metric line format:
  INFO <ts> ot_train.py:769 step:200 smpl:400 ep:2 epch:0.00 loss:0.032 grdn:0.086 lr:2.7e-06 ...
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW_RE = re.compile(r"ot_train\.py:\d+\s+step:([\d.]+[KkM]?)([^\n]*)")


def parse(path, log_freq=100):
    """Extract metric rows. lerobot abbreviates step>=1000 as e.g. '4K' (rounded,
    ambiguous), but rows are emitted strictly every `log_freq` steps in order,
    so the true step is recovered from the row index."""
    rows = []
    content = Path(path).read_text(errors="replace").replace("\r", "\n")
    for m in ROW_RE.finditer(content):
        row = {}
        for kv in m.group(2).split():
            if ":" in kv:
                k, v = kv.split(":", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        if "loss" in row:
            row["step"] = (len(rows) + 1) * log_freq
            rows.append(row)
    return rows


def plot(rows, key, ylabel, out, log_scale=False):
    xs = [r["step"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    if not xs:
        return False
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs step")
    plt.grid(alpha=0.3)
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return True


def main(log_path, out_dir):
    rows = parse(log_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no metric rows yet")
        return
    keys = sorted({k for r in rows for k in r} - {"step"})
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"parsed {len(rows)} rows -> metrics.csv (keys: {keys})")

    specs = [
        ("loss", "training loss", True),
        ("lr", "learning rate", False),
        ("grdn", "grad norm", True),
        ("smp/s", "throughput (samples/s)", False),
        ("mem_gb", "GPU memory (GB)", False),
        ("epch", "epoch", False),
    ]
    for key, label, logsc in specs:
        if plot(rows, key, label, out / f"{key.replace('/', '_')}.png", logsc):
            print(f"  plot: {label}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
