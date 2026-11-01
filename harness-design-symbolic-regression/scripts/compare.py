"""Paired comparison of two harnesses from <out_dir>/results.

    uv run python scripts/compare.py --out runs/reasoning_ablation \
        baseline_autoresearch_no_reasoning baseline_autoresearch
"""
import argparse

import pandas as pd

from sr_harness.analysis.paired import paired_table
from sr_harness.analysis.plots import load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harness_a")
    ap.add_argument("harness_b")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    results = load_results(args.out)
    table, n = paired_table(results, args.harness_a, args.harness_b)
    seeds = sorted({r["seed"] for r in results})
    print(f"A = {args.harness_a}   B = {args.harness_b}   n = {n} tasks, seeds {seeds} (averaged per task)\n")
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
        print(table.to_string())


if __name__ == "__main__":
    main()
