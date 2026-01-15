import json
from pathlib import Path
import numpy as np

ROOT = Path("data")

import json

def load_jsonl(path, *, skip_bad=True, verbose=True):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    preview = line[:200].replace("\n", "\\n")
                    print(f"[WARN] {path} line {lineno}: JSONDecodeError: {e}. Line starts: {preview!r}")
                if not skip_bad:
                    raise
                continue

def summarize_experiment(jsonl_path):
    total_tokens = 0
    total_time = 0.0
    accept_lengths = []

    for record in load_jsonl(jsonl_path):
        for choice in record["choices"]:
            total_tokens += sum(choice.get("new_tokens", []))
            total_time += sum(choice.get("wall_time", []))
            accept_lengths.extend(choice.get("accept_lengths", []))

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    mean_accept = np.mean(accept_lengths) if accept_lengths else 0.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "mean_accept": mean_accept,
    }

for benchmark_dir in ROOT.iterdir():
    model_answer_dir = benchmark_dir / "model_answer"
    if not model_answer_dir.exists():
        continue

    print(f"\n=== Benchmark: {benchmark_dir.name} ===")

    results = {}

    for jsonl_file in model_answer_dir.glob("*.jsonl"):
        stats = summarize_experiment(jsonl_file)
        results[jsonl_file.stem] = stats

    # detect vanilla baseline
    vanilla_key = next(
        (k for k in results if "vanilla" in k),
        None
    )
    vanilla_tps = results[vanilla_key]["tokens_per_sec"] if vanilla_key else None

    for name, stats in results.items():
        speedup = (
            stats["tokens_per_sec"] / vanilla_tps
            if vanilla_tps and "vanilla" not in name
            else 1.0 if name == vanilla_key else float("nan")
        )

        print(
            f"{name:55s} | "
            f"tokens/sec: {stats['tokens_per_sec']:.2f} | "
            f"mean_accept: {stats['mean_accept']:.2f} | "
            f"speedup: {speedup:.2f}"
        )
