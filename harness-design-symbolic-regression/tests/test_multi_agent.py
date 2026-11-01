import pytest

from fakes import FakeLLM
from sr_harness import load_config
from sr_harness.analysis import multi_agent as MA
from sr_harness.benchmark import toy_tasks
from sr_harness.experiments.runner import run_one


@pytest.mark.parametrize("k", [1, 2, 4])
def test_rounds_and_workers_from_log(k, tmp_path):
    cfg = load_config()
    cfg["experiment"]["budget"].update(max_evaluations=8, max_model_calls=200)
    rec = run_one(toy_tasks()[0], f"plan_execute_k{k}", 0, cfg, lambda logger, seed: FakeLLM(), tmp_path)
    rds = MA.rounds(MA.log_path(tmp_path, rec))
    assert len(rds) == 8 // k                              # 8 evaluations, one per worker, k workers per round
    assert all(len(rd["workers"]) == k for rd in rds)
    assert all(len(w["evaluations"]) == 1 for rd in rds for w in rd["workers"])
    assert rec["evaluations"] == 8
    per_round = MA.round_table([rec], tmp_path, "ood_r2")
    assert list(per_round["round"]) == list(range(1, len(rds) + 1))


def test_family_and_equivalence():
    assert MA.family("c0*x**2 + c1") == "polynomial"
    assert MA.family("c0*sin(t) + c1*exp(x)") == "trig+exp/log"
    assert MA.family("c0/(x + c1)") == "rational"
    task = toy_tasks()[0]
    a = {"expression": "c0*x**2", "params": {"c0": 2.0}}
    b = {"expression": "2*x*x", "params": {}}
    c = {"expression": "c0*x", "params": {"c0": 2.0}}
    assert MA.equivalent(a, b, task) and not MA.equivalent(a, c, task)


def test_canonical_form_matches_up_to_order_and_constant_names():
    v = ["x", "t"]
    same = lambda a, b: MA.canonical_form(a, v) == MA.canonical_form(b, v)
    assert same("c0*x + c1", "c1 + x*c0")                 # order of terms and factors
    assert same("c0*x + c1", "c3 + c2*x")                 # names of the constants
    assert same("c0*x^2 + c1*sin(t)", "c1*sin(t) + c0*x**2")
    assert not same("c0*sin(t)", "c0*cos(t)")
    assert not same("c0*x + c1", "c0*x + c1*t")
    assert not same("c0*(x + 1)", "c0*x + c1")            # an algebraic rewrite is a different form
