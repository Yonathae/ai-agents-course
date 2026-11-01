"""Paired comparison of two harnesses on the same tasks (as in Lecture 4).

For each task i, average each metric over seeds, then take the difference
D_i = m_B(i) - m_A(i). Report mean(D) with a 95% normal confidence interval,
mean(D) +- 1.96 * sd(D) / sqrt(n), over the n tasks both harnesses ran.
MSE is compared as log10 MSE (lower is better; a difference of -0.3 is about 2x smaller).
"""
import math

import pandas as pd

from .plots import final_log10_mse


def _r2(split):
    return lambda r: max(0.0, r["scores"][split]["r2"]) if r["scores"][split] else 0.0


METRICS = {   # name -> how to read it from one result record
    "train R² (clipped at 0)": lambda r: max(0.0, r["best_train_r2"]) if r["best_train_r2"] is not None else 0.0,
    "ID R² (clipped at 0)": _r2("train_dist"),
    "OOD R² (clipped at 0)": _r2("ood"),
    "log10 train MSE": lambda r: final_log10_mse(r, "train"),
    "log10 OOD MSE": lambda r: final_log10_mse(r, "ood"),
    "evaluations used": lambda r: r["evaluations"],
    "model calls": lambda r: r["model_calls"],
    "tokens": lambda r: r["input_tokens"] + r["output_tokens"],
    "cost ($)": lambda r: r["cost"],
}


def per_task(results, harness):
    rows = [dict(task=r["task_id"], seed=r["seed"], **{k: f(r) for k, f in METRICS.items()})
            for r in results if r["harness"] == harness]
    return pd.DataFrame(rows).groupby("task").mean(numeric_only=True).drop(columns="seed")


def paired_table(results, harness_a, harness_b):
    a, b = per_task(results, harness_a), per_task(results, harness_b)
    tasks = a.index.intersection(b.index)
    a, b = a.loc[tasks], b.loc[tasks]
    rows = []
    for m in METRICS:
        d = b[m] - a[m]
        n = len(d)
        half = 1.96 * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        rows.append({"metric": m, harness_a: a[m].mean(), harness_b: b[m].mean(),
                     "diff (B - A)": d.mean(), "95% CI": f"[{d.mean() - half:+.3g}, {d.mean() + half:+.3g}]",
                     "median diff": d.median(),
                     "tasks B higher / lower / tie": f"{(d > 1e-9).sum()} / {(d < -1e-9).sum()} / "
                                                     f"{(d.abs() <= 1e-9).sum()}"})
    return pd.DataFrame(rows).set_index("metric"), len(tasks)
