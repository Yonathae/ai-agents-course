"""Analysis for the plan-and-execute questions (slide 17), from run logs.

  rounds(log)            the run as rounds -> workers -> evaluated equations
  family(expr)           a coarse equation family: polynomial, trig, exp/log, rational, or a mix
  equivalent(a, b, X)    do two fitted equations make the same predictions on X?
  canonical_form(expr)   the equation up to the order of terms and the names of its constants
  diversity_table        per round with k >= 2 workers: distinct families, equivalent worker pairs
  plot_vs_rounds         R^2 or MSE vs. rounds (parallel time), each harness up to its own last round

Workers run one after another, so every evaluation in the log belongs to the most recent
worker_start (which records its round, worker index and direction). Each worker proposes one
equation, so each worker has (at most) one evaluation.
"""
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..benchmark.evaluator import ExpressionError, parse_expression, predict
from ..experiments.logging import read_log
from . import plots


def rounds(log_path):
    """[{round, workers: [{worker, direction, evaluations: [evaluation event, ...]}]}]."""
    out, cur = [], None
    for e in read_log(log_path):
        if e["event"] == "worker_start":
            if not out or out[-1]["round"] != e["round"]:
                out.append({"round": e["round"], "workers": []})
            cur = {"worker": e["worker"], "direction": e["direction"], "evaluations": []}
            out[-1]["workers"].append(cur)
        elif e["event"] == "evaluation" and cur is not None:
            cur["evaluations"].append(e)
    return out


def log_path(out_dir, r):
    return Path(out_dir, "logs", f"{r['run_id']}.jsonl")


FAMILY_PATTERNS = {"trig": r"\b(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh)\(",
                   "exp/log": r"\b(exp|log)\(|\be\*\*", "sqrt/abs": r"\b(sqrt|abs)\(",
                   "rational": r"/\s*\(|/\s*[A-Za-z]"}


def family(expr):
    """Coarse family from the functions an expression uses; 'polynomial' if none."""
    found = [f for f, pat in FAMILY_PATTERNS.items() if re.search(pat, expr or "")]
    return "+".join(found) if found else "polynomial"


def equivalent(a, b, task, rtol=1e-3):
    """Same predictions on the training inputs (relative RMS difference < rtol)?"""
    try:
        ya = predict(parse_expression(a["expression"], task.variable_names, 99, 10_000)[0],
                     task.variable_names, a["params"], task.x_train)
        yb = predict(parse_expression(b["expression"], task.variable_names, 99, 10_000)[0],
                     task.variable_names, b["params"], task.x_train)
    except ExpressionError:
        return False
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(yb))):
        return False
    return float(np.sqrt(np.mean((ya - yb) ** 2)) / (np.std(task.y_train) + 1e-300)) < rtol


SYMPY_FUNCTIONS = {"sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin", "arccos": "acos", "arctan": "atan",
                   "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "log", "sqrt": "sqrt",
                   "abs": "Abs"}


def canonical_form(expr, variable_names):
    """The equation as a canonical string: sympy orders the terms and factors, and every free constant
    c0, c1, ... becomes the same symbol C. Two equations with the same canonical form are the same
    equation up to reordering and renaming of constants (their fitted values do not matter)."""
    import sympy
    parse_expression(expr, variable_names, 99, 10_000)            # validate first: never sympify raw text
    text = expr.strip().replace("^", "**").split("=")[-1]
    names = {v: sympy.Symbol(v) for v in variable_names}
    names.update({f: getattr(sympy, g) for f, g in SYMPY_FUNCTIONS.items()})
    names.update({"pi": sympy.pi, "e": sympy.E})
    names.update({f"c{i}": sympy.Symbol(f"c{i}") for i in range(100)})
    e = sympy.sympify(text, locals=names)
    C = sympy.Symbol("C")
    e = e.subs({c: C for c in e.free_symbols if re.fullmatch(r"c\d+", c.name)})
    return sympy.srepr(e)


def diversity_table(results, out_dir, tasks):
    """One row per (run, round) with k >= 2 workers: how many workers, how many distinct families
    among their equations, the fraction of worker pairs with equivalent predictions, and whether two
    workers proposed the same equation (same canonical form)."""
    rows = []
    for r in results:
        task = tasks[r["task_id"]]
        for rd in rounds(log_path(out_dir, r)):
            best = [max((e for e in w["evaluations"] if e["error"] is None), key=lambda e: e["train_r2"],
                        default=None) for w in rd["workers"]]
            best = [b for b in best if b is not None]
            if len(best) < 2:
                continue
            pairs = list(itertools.combinations(best, 2))
            forms = [canonical_form(b["expression"], task.variable_names) for b in best]
            rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], round=rd["round"],
                             workers=len(best), families=len({family(b["expression"]) for b in best}),
                             equivalent_pairs=np.mean([equivalent(a, b, task) for a, b in pairs]),
                             same_form=len(set(forms)) < len(forms)))
    return pd.DataFrame(rows)


def round_table(results, out_dir, metric):
    """One row per (run, round): the run's best-so-far score after that round. A run that
    stopped early keeps its last score until its budget's last round."""
    rows = []
    for r in results:
        rds = rounds(log_path(out_dir, r))
        cum = list(np.cumsum([sum(len(w["evaluations"]) for w in rd["workers"]) for rd in rds])) or [0]
        curve = sorted(r["curve"], key=lambda c: c["index"])
        for i, n in enumerate(cum, 1):
            seen = [c for c in curve if c["index"] <= n]
            rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], round=i,
                             value=plots.value(seen[-1] if seen else None, metric, r)))
    return pd.DataFrame(rows)


def plot_vs_rounds(results, out_dir, metric, last_round, ideal=None, ax=None):
    """Mean over tasks (seeds averaged per task) with 95% bootstrap bands; harness h is drawn up to
    round last_round[h] (its budget); runs that ended earlier keep their last score.
    ideal = (base, {h: k}): for each h, a dotted line at the base harness's (one worker's) mean at
    round k·r, i.e. a perfect k-fold speedup."""
    df = round_table(results, out_dir, metric)
    ax = ax or plt.gca()

    def mean_ci(h, x):
        # a run that ended before round x keeps its last value
        d = df[(df.harness == h) & (df["round"] <= x)]
        per_task = d.sort_values("round").groupby(["task", "seed"]).tail(1).groupby("task")["value"].mean()
        return (per_task.mean(), *plots.bootstrap_ci(per_task.values))

    for h, last in last_round.items():
        if df[df.harness == h].empty:
            continue
        xs = list(range(1, last + 1))
        mean, lo, hi = zip(*(mean_ci(h, x) for x in xs))
        ax.plot(xs, plots.back(mean, metric), color=plots.COLORS.get(h), ls=plots.linestyle(h), label=h)
        ax.fill_between(xs, plots.back(lo, metric), plots.back(hi, metric), color=plots.COLORS.get(h),
                        alpha=0.12, lw=0)
    if ideal:
        base, ks = ideal
        for h, k in ks.items():
            xs = list(range(1, last_round[h] + 1))
            ys = [mean_ci(base, k * x)[0] for x in xs]
            ax.plot(xs, plots.back(ys, metric), color=plots.COLORS.get(h), ls=":", lw=1.5, label=f"ideal for {h}")
    ax.set(xscale="log", yscale="log" if plots.is_log(metric) else "linear", xlabel="round",
           ylabel=plots.METRICS[metric])
    ax.legend(frameon=False, fontsize=8)
    return ax
