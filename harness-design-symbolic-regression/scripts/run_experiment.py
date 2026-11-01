"""Run the harness comparison.

    uv run python scripts/run_experiment.py --toy --harness baseline_autoresearch --seeds 0 --max-evaluations 4
    uv run python scripts/run_experiment.py                      # full grid from configs/

Results: <out_dir>/results/*.json (one per run), logs: <out_dir>/logs/*.jsonl.
Finished runs are skipped, so an interrupted grid can be resumed.
"""
import argparse

from sr_harness import load_config
from sr_harness.benchmark import load_tasks, toy_tasks
from sr_harness.experiments.runner import run_grid
from sr_harness.llm import LLMClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="offline toy tasks instead of LLM-SRBench")
    ap.add_argument("--harness", nargs="*", help="default: all harnesses in experiment.yaml")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--n-tasks", type=int)
    ap.add_argument("--max-evaluations", type=int)
    ap.add_argument("--max-model-calls", type=int, help="safety stop (default: experiment.yaml)")
    ap.add_argument("--out", help="default: experiment.yaml out_dir")
    ap.add_argument("--shard", default="0/1", help="k/n: run every n-th task starting at k (parallel processes)")
    args = ap.parse_args()

    cfg = load_config()
    if args.n_tasks:
        cfg["benchmark"]["n_tasks"] = args.n_tasks
    if args.max_evaluations:
        cfg["experiment"]["budget"]["max_evaluations"] = args.max_evaluations
    if args.max_model_calls:
        cfg["experiment"]["budget"]["max_model_calls"] = args.max_model_calls
    tasks = toy_tasks() if args.toy else load_tasks(cfg["benchmark"])
    k, n = map(int, args.shard.split("/"))
    tasks = tasks[k::n]
    harnesses = args.harness or list(cfg["experiment"]["harnesses"])
    seeds = args.seeds if args.seeds is not None else cfg["experiment"]["seeds"]

    make_llm = lambda logger, seed: LLMClient(cfg["model"], logger=logger, seed=seed)
    run_grid(tasks, harnesses, seeds, cfg, make_llm, args.out or cfg["experiment"]["out_dir"])


if __name__ == "__main__":
    main()
