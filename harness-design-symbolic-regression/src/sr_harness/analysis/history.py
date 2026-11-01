"""Analysis for the context-management question (slide 20), from run logs.

  calls_by_step(out_dir, results)   every model call: harness, role, time step, input tokens
  repeats(out_dir, results, tasks)  per run: evaluations, and how many re-propose an equation
                                    the run already tried (same equation up to order and constant names)
"""
from pathlib import Path

import pandas as pd

from ..experiments.logging import read_log
from .multi_agent import canonical_form


def _log(out_dir, r):
    return Path(out_dir, "logs", f"{r['run_id']}.jsonl")


def calls_by_step(out_dir, results):
    """One row per model call. Its time step is the number of equations evaluated before it, plus one."""
    rows = []
    for r in results:
        step = 1
        for e in read_log(_log(out_dir, r)):
            if e["event"] == "model_call":
                rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], role=e["role"],
                                 step=step, input_tokens=e["input_tokens"]))
            elif e["event"] == "evaluation":
                step += 1
    return pd.DataFrame(rows)


def repeats(out_dir, results, tasks):
    """One row per run: valid evaluations, and how many of them repeat an earlier equation of the run."""
    rows = []
    for r in results:
        names = tasks[r["task_id"]].variable_names
        seen, n, rep = set(), 0, 0
        for e in read_log(_log(out_dir, r)):
            if e["event"] == "evaluation" and e["error"] is None:
                form = canonical_form(e["expression"], names)
                n += 1
                rep += form in seen
                seen.add(form)
        rows.append(dict(harness=r["harness"], task=r["task_id"], seed=r["seed"], evaluations=n, repeats=rep))
    return pd.DataFrame(rows)
