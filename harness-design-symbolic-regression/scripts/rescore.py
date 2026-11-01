"""Recompute the hidden scores and curves of finished runs from their logs (no model calls).

    uv run python scripts/rescore.py runs/reasoning_ablation runs/pilot2
"""
import sys

from sr_harness import load_config
from sr_harness.benchmark.loader import load_llm_srbench, toy_tasks
from sr_harness.experiments.runner import rescore_results


def main():
    cfg = load_config()
    tasks = load_llm_srbench(cfg["benchmark"]["subsets"], cfg["benchmark"]["data_dir"]) + toy_tasks()
    for out_dir in sys.argv[1:]:
        print(f"{out_dir}: rescored {rescore_results(out_dir, tasks, cfg)} runs")


if __name__ == "__main__":
    main()
