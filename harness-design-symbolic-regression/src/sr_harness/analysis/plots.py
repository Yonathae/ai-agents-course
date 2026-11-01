"""The lecture plots, from results/*.json.

  plot_vs_steps          performance vs. time step (equation evaluations)
  plot_vs_tokens         performance vs. cumulative tokens (fixed token budget)
  final_log10_mse        a run's final MSE (log10), for the paired comparison

Metrics (curve fields), see METRICS: R^2 and MSE on train (what the harness optimizes),
ID (new points in the training range) and OOD (beyond the range).
MSE is in the target's units, so train and OOD of one task are directly comparable. Across
tasks it is averaged in log space (a geometric mean): target scales differ by up to 16 orders
of magnitude, and an arithmetic mean would be the MSE of the largest-scale task. Before the
first valid equation a run's MSE is that of predicting the training mean. R^2 is clipped at 0
and averaged directly.
Averaged over tasks (seeds averaged per task), with 95% bootstrap intervals over tasks.
Style follows the course decks: grayscale + one accent, black error bars.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"baseline_autoresearch": "#222222",
          "baseline_autoresearch_no_reasoning": "#2a9fd6", "plan_execute": "#c0392b",
          "context_managed": "#2a9fd6",
          # H2 with k workers: one color per k; the diverse ablation uses the same color, dashed
          "plan_execute_k1": "#222222", "plan_execute_k2": "#c0392b", "plan_execute_k4": "#2a9fd6",
          "plan_execute_k2_diverse": "#c0392b", "plan_execute_k4_diverse": "#2a9fd6"}


def linestyle(h):
    return "--" if h.endswith("_diverse") else "-"

METRICS = {   # curve field -> axis label
    "train_r2": "train R² (clipped at 0)",
    "train_dist_r2": "ID R² (clipped at 0)",
    "ood_r2": "OOD R² (clipped at 0)",
    "train_mse": "train MSE (geometric mean over tasks)",
    "train_dist_mse": "ID MSE (geometric mean over tasks)",
    "ood_mse": "OOD MSE (geometric mean over tasks)",
    "train_nmse": "train NMSE (MSE / var of training targets)",
    "ood_nmse": "OOD NMSE (MSE / var of training targets)",
    "tokens": "cumulative tokens",
}


def load_results(out_dir):
    return [json.load(open(p)) for p in sorted(Path(out_dir, "results").glob("*.json"))]


def is_log(metric):
    """Squared errors (MSE, NMSE) are averaged in log space and shown on a log axis."""
    return metric.endswith("mse")


def log10_err(v, cap):
    """log10 of a squared error; `cap` (already log10) for a missing or diverging value."""
    if v is None or not np.isfinite(float(v)):
        return cap
    return float(min(cap, np.log10(max(float(v), 1e-300))))


def empty_value(r, metric):
    """The score of run r before (or without) any valid equation: R^2 = 0, and for squared
    errors the error of predicting the training mean (log10)."""
    if metric == "tokens":
        return 0.0
    if not is_log(metric):
        return 0.0
    split = metric.rsplit("_", 1)[0]                       # train_mse -> train, ood_nmse -> ood
    mse = r["baseline_mse"][split]
    return float(np.log10(mse / r["target_var"] if metric.endswith("nmse") else mse))


def value(c, metric, r):
    """A curve entry's value: R^2 clipped at 0, squared errors as log10, tokens as is;
    c = None (no valid equation yet) gives empty_value."""
    v = c.get(metric) if c else None
    if metric == "tokens":
        return float(c["tokens"]) if c else 0.0
    if v is None:
        return empty_value(r, metric)
    if is_log(metric):
        return log10_err(v, cap=300.0)
    return max(0.0, float(v))


def back(x, metric):
    """Undo the log for squared errors so the axis shows the error itself."""
    return 10.0 ** np.asarray(x) if is_log(metric) else np.asarray(x)


def bootstrap_ci(per_task, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(per_task, dtype=float)
    means = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def step_table(results, metric, max_step):
    """One row per (harness, task, seed, step k): the run's best-so-far value after k evaluations.
    A run that stopped early keeps its last value (its answer does not change)."""
    rows = []
    for r in results:
        curve = sorted(r["curve"], key=lambda c: c["index"])
        for k in range(1, max_step + 1):
            seen = [c for c in curve if c["index"] <= k]
            rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], step=k,
                             value=value(seen[-1] if seen else None, metric, r)))
    return pd.DataFrame(rows)


def _errorbar_lines(ax, df, xs, xcol, metric, log=False):
    for i, h in enumerate(sorted(df.harness.unique())):
        means, lo, hi = [], [], []
        for x in xs:
            per_task = df[(df.harness == h) & (df[xcol] == x)].groupby("task")["value"].mean()
            means.append(per_task.mean())
            l, u = bootstrap_ci(per_task.values)
            lo.append(l), hi.append(u)
        means, lo, hi = back(means, metric), back(lo, metric), back(hi, metric)
        off = np.array(xs) * (1 + (i - 0.5) * 0.03) if log else np.array(xs) + (i - 0.5) * 0.06
        ax.errorbar(off, means, yerr=[means - lo, hi - means], color=COLORS.get(h), ls=linestyle(h),
                    marker="o", capsize=3, ecolor="black", elinewidth=0.8, label=h)
    if is_log(metric):
        ax.set_yscale("log")
    ax.set(xticks=xs, xticklabels=xs, ylabel=METRICS[metric])
    ax.legend(frameon=False)
    return ax


def plot_vs_steps(results, metric="ood_r2", max_step=None, ax=None):
    """Performance vs. time step (evaluation index)."""
    max_step = max_step or max(c["index"] for r in results for c in r["curve"])
    ax = ax or plt.gca()
    _errorbar_lines(ax, step_table(results, metric, max_step), list(range(1, max_step + 1)), "step", metric)
    ax.set_xlabel("time step (equation evaluations)")
    return ax


def tokens_table(results, metric, grid):
    """One row per (harness, task, seed, token budget g): the run's best-so-far score within g
    tokens; empty_value before its first evaluation. A run that ended keeps its last score."""
    rows = []
    for r in results:
        curve = sorted(r["curve"], key=lambda c: c["index"])
        for g in grid:
            seen = [c for c in curve if c["tokens"] <= g]
            rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], tokens=g,
                             value=value(seen[-1] if seen else None, metric, r)))
    return pd.DataFrame(rows)


def plot_vs_tokens(results, n_grid=20, metric="ood_r2", ax=None):
    """Performance at a fixed token budget (all model calls), with a 95% bootstrap band over tasks."""
    top = max((c["tokens"] for r in results for c in r["curve"]), default=1)
    grid = list(np.geomspace(1000, max(1000, top), n_grid))
    df = tokens_table(results, metric, grid)
    ax = ax or plt.gca()
    for h in sorted(df.harness.unique()):
        mean, lo, hi = [], [], []
        for g in grid:
            per_task = df[(df.harness == h) & (df.tokens == g)].groupby("task")["value"].mean()
            mean.append(per_task.mean())
            l, u = bootstrap_ci(per_task.values)
            lo.append(l), hi.append(u)
        ax.plot(grid, back(mean, metric), color=COLORS.get(h), ls=linestyle(h), label=h)
        ax.fill_between(grid, back(lo, metric), back(hi, metric), color=COLORS.get(h), alpha=0.15, lw=0)
    ax.set(xscale="log", yscale="log" if is_log(metric) else "linear",
           xlabel="cumulative inference tokens (all roles)", ylabel=METRICS[metric])
    ax.legend(frameon=False)
    return ax


def final_log10_mse(r, split):
    """log10 MSE of a run's final equation on a split (train: from its curve). No equation at all:
    the error of predicting the training mean; a diverging equation: capped as in the runner."""
    if split == "train":
        mse = r["curve"][-1]["train_mse"] if r["curve"] else None
    else:
        mse = r["scores"][split]["mse"] if r["scores"][split] else None
    if mse is None:
        return empty_value(r, f"{split}_mse")
    return log10_err(mse, cap=float(np.log10(1e12 * r["target_var"])))
