import time

import numpy as np
import pytest

from sr_harness import load_config
from sr_harness.benchmark import Evaluator, toy_tasks
from sr_harness.evaluation.metrics import score_equation

CFG = load_config()["benchmark"]["evaluator"]


@pytest.fixture
def task():
    return toy_tasks()[0]          # y = 0.8 x^2 - 1.5 sin(t)


@pytest.fixture
def ev(task):
    return Evaluator(task.view(), CFG)


def test_true_form_fits_constants(ev):
    r = ev.evaluate("c0*x**2 + c1*sin(t)")
    assert r.ok and r.train_r2 > 0.999999
    assert abs(r.params["c0"] - 0.8) < 1e-6 and abs(r.params["c1"] + 1.5) < 1e-6


def test_wrong_form_scores_lower(ev):
    assert ev.evaluate("c0*x + c1").train_r2 < 0.5


def test_caret_and_lhs_are_accepted(ev):
    assert ev.evaluate("y = c0*x^2 + c1*sin(t)").train_r2 > 0.999999


def test_deterministic(ev):
    a, b = ev.evaluate("c0*exp(c1*x) + c2*t"), ev.evaluate("c0*exp(c1*x) + c2*t")
    assert a.params == b.params and a.train_r2 == b.train_r2


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "x.__class__", "open('f')", "[x for x in t]",
    "lambda: 1", "z + 1", "c0(x)", "sin(x, out=t)", "x if t else 1",
])
def test_unsafe_or_invalid_rejected(ev, expr):
    r = ev.evaluate(expr)
    assert not r.ok and r.error


def test_syntax_error(ev):
    assert "syntax" in ev.evaluate("c0*x +* ").error


def test_non_finite(ev):
    assert "non-finite" in ev.evaluate("1/(x-x)").error
    assert not ev.evaluate("log(-1 - x**2)").ok


def test_huge_power_does_not_hang(ev):
    t0 = time.time()
    r = ev.evaluate("9**9**9 + x")
    assert not r.ok and time.time() - t0 < 1


def test_too_many_params(ev):
    expr = " + ".join(f"c{i}*x" for i in range(CFG["max_params"] + 1))
    assert "too many" in ev.evaluate(expr).error


def test_hidden_scoring_uses_train_constants(task, ev):
    r = ev.evaluate("c0*x**2 + c1*sin(t)")                  # the true form
    for split in ("train_dist", "ood"):
        s = score_equation(task, r.expression, r.params, split)
        assert s["valid"] and s["nmse"] < 1e-6 and s["r2"] > 0.9999
    wrong = ev.evaluate("c0*x + c1*t + c2")                 # a wrong form
    s = score_equation(task, wrong.expression, wrong.params, "ood")
    assert s["r2"] < 0.9 and s["nmse"] > 0.01


def test_curve_fit_fits_the_range_but_fails_beyond_it():
    task = toy_tasks()[1]                                   # P = 2 t exp(-t/2), train t in [0, 4]
    ev = Evaluator(task.view(), CFG)
    poly = ev.evaluate("c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4")
    assert poly.train_r2 > 0.999
    assert score_equation(task, poly.expression, poly.params, "train_dist")["r2"] > 0.999
    assert score_equation(task, poly.expression, poly.params, "ood")["r2"] < 0.5
