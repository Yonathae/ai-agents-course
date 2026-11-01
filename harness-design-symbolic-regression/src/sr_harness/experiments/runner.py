"""Run harnesses on tasks and score the results on hidden data.

One run = (task, harness, seed). The runner is the only place that holds the full
task: it hands the harness a TaskView, and scores on held-out data only after the run.

Three scores per equation:
  train R^2               on the data the agent searches on (what the harness optimizes)
  ID R^2  new points from the training range (hidden)
  OOD R^2                points beyond the training range (hidden): did it find the law?
plus MSE (target units) and NMSE (MSE / var(y_train)) on train and on both hidden splits.

Anytime curves: after each evaluation k, the run's answer is the best-so-far
equation by train R^2 (the same selection rule the harness uses at the end).
Its hidden scores at k give performance vs. time step, and the cumulative tokens
at k give performance vs. tokens, all from one run at the maximum budget.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..benchmark.evaluator import Evaluator
from ..evaluation.holdout import holdout_scores
from ..harnesses import ResearchSession, make_harness
from .budget import Budget
from .logging import RunLogger, read_log


def run_one(task, harness_name, seed, cfg, make_llm, out_dir):
    """make_llm(logger, seed) -> an LLMClient (or a fake with the same interface)."""
    exp, bench = cfg["experiment"], cfg["benchmark"]
    run_id = f"{harness_name}__{task.task_id.replace('/', '-')}__s{seed}"
    out_dir = Path(out_dir)
    logger = RunLogger(out_dir / "logs" / f"{run_id}.jsonl",
                       dict(run_id=run_id, task_id=task.task_id, harness=harness_name, seed=seed,
                            model=cfg["model"]["model"], budget=exp["budget"]))
    view = task.view()                                   # no hidden data from here on
    session = ResearchSession(view, Evaluator(view, bench["evaluator"]), make_llm(logger, seed),
                              Budget(**exp["budget"]), logger, bench["prompt"], bench["evaluator"]["max_params"])
    harness = make_harness(harness_name, exp["harnesses"][harness_name])
    try:
        result = harness.run(session)
        status = "ok"
    except Exception as e:                                # keep the grid going; record the failure
        logger.log("run_error", error=repr(e))
        result, status = session.result(harness_name), f"error: {e!r}"
    logger.log("final", **asdict(result))
    logger.close()

    record = dict(run_id=run_id, seed=seed, status=status, **asdict(result))
    record.update(score_record(task, record, logger.path, bench["scoring"]))
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    json.dump(record, open(out_dir / "results" / f"{run_id}.json", "w"), indent=1, default=str)
    return record


MSE_CAP = 1e12      # a diverging equation counts as MSE = MSE_CAP * var(y_train)


def score_record(task, record, log_path, scoring_cfg):
    """The hidden scores of a run's final equation and its anytime curve."""
    return {"scores": holdout_scores(task, record["best_expression"], record["best_params"], scoring_cfg),
            "curve": anytime_curve(task, log_path, scoring_cfg),
            "target_var": float(np.var(task.y_train)),
            # the score of a run (or step) without any valid equation: predict the training mean
            "baseline_mse": {s: baseline_mse(task, s) for s in ("train", "train_dist", "ood")}}


def baseline_mse(task, split):
    """MSE of predicting the training mean: the score before the first valid equation."""
    y = {"train": task.y_train, "train_dist": task.y_train_dist, "ood": task.y_ood}[split]
    return float(np.mean((y - task.y_train.mean()) ** 2))


def anytime_curve(task, log_path, scoring_cfg):
    var = float(np.var(task.y_train))
    cap = lambda mse, split: baseline_mse(task, split) if mse is None else min(float(mse), MSE_CAP * var)
    curve, best, s = [], None, None
    for ev in read_log(log_path):
        if ev["event"] != "evaluation":
            continue
        if ev["error"] is None and (best is None or ev["train_r2"] > best["train_r2"]):
            best = ev
            s = holdout_scores(task, best["expression"], best["params"], scoring_cfg)
        get = lambda split, key: s[split][key] if s and s[split] else None
        curve.append(dict(index=ev["index"], tokens=ev["tokens"], cost=ev["cost"], role=ev["role"],
                          best_expression=best and best["expression"], train_r2=best and best["train_r2"],
                          train_mse=cap(best and best["train_nmse"] * var, "train"),
                          train_nmse=best and best["train_nmse"],
                          train_dist_r2=get("train_dist", "r2"),
                          train_dist_mse=cap(get("train_dist", "mse"), "train_dist"),
                          train_dist_nmse=get("train_dist", "nmse"),
                          ood_r2=get("ood", "r2"), ood_mse=cap(get("ood", "mse"), "ood"),
                          ood_nmse=get("ood", "nmse")))
    return curve


def rescore_results(out_dir, tasks, cfg):
    """Recompute scores and curves of existing results from their logs (no model calls),
    e.g. after the metrics change."""
    by_id = {t.task_id: t for t in tasks}
    n = 0
    for p in sorted(Path(out_dir, "results").glob("*.json")):
        rec = json.load(open(p))
        task = by_id.get(rec["task_id"])
        if task is None:
            continue
        rec.update(score_record(task, rec, Path(out_dir, "logs", p.stem + ".jsonl"), cfg["benchmark"]["scoring"]))
        json.dump(rec, open(p, "w"), indent=1, default=str)
        n += 1
    return n


def run_grid(tasks, harness_names, seeds, cfg, make_llm, out_dir, skip_existing=True):
    out = []
    for task in tasks:
        for name in harness_names:
            for seed in seeds:
                run_id = f"{name}__{task.task_id.replace('/', '-')}__s{seed}"
                path = Path(out_dir) / "results" / f"{run_id}.json"
                if skip_existing and path.exists():
                    out.append(json.load(open(path)))
                    continue
                rec = run_one(task, name, seed, cfg, make_llm, out_dir)
                print(f"{run_id:60s} evals={rec['evaluations']:3d} tokens={rec['input_tokens'] + rec['output_tokens']:7d}"
                      f" ID R^2={_fmt(rec['scores']['train_dist'])}"
                      f" OOD R^2={_fmt(rec['scores']['ood'])}  {rec['status']}")
                out.append(rec)
    return out


def _fmt(s):
    return f"{s['r2']:.4f}" if s else "n/a"
