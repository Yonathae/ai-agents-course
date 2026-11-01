"""Held-out scores of one final equation, reported separately:

  train_dist  new points from the training range: does the fit interpolate?
  ood         points beyond the training range (the benchmark's predefined split,
              never a shift we invent after seeing results): did the search find
              the law, or only a curve that fits the observed range?
"""
from .metrics import score_equation


def holdout_scores(task, expression, params, cfg=None):
    return {"train_dist": score_equation(task, expression, params, "train_dist", cfg),
            "ood": score_equation(task, expression, params, "ood", cfg)}
