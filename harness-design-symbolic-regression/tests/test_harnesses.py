import dataclasses
import json

import pytest

from fakes import FakeLLM
from sr_harness import load_config
from sr_harness.benchmark import TaskView, toy_tasks
from sr_harness.experiments.runner import run_one


@pytest.fixture
def cfg():
    c = load_config()
    c["experiment"]["budget"]["max_evaluations"] = 7     # not a multiple of the per-investigation share
    return c


def _run(name, cfg, tmp_path, fake=None):
    fake = fake or FakeLLM()
    rec = run_one(toy_tasks()[0], name, 0, cfg, lambda logger, seed: fake, tmp_path)
    return rec, fake


@pytest.mark.parametrize("name", ["baseline_autoresearch", "baseline_autoresearch_no_reasoning", "plan_execute", "context_managed"])
def test_budget_and_scoring(name, cfg, tmp_path):
    rec, _ = _run(name, cfg, tmp_path)
    assert rec["status"] == "ok"
    assert rec["evaluations"] == 7
    assert len(rec["curve"]) == 7 and [c["index"] for c in rec["curve"]] == list(range(1, 8))
    assert rec["best_expression"] is not None and rec["scores"]["train_dist"]["valid"] and rec["scores"]["ood"]["valid"]
    assert rec["curve"][-1]["tokens"] <= rec["input_tokens"] + rec["output_tokens"]


ALL = ["baseline_autoresearch", "baseline_autoresearch_no_reasoning", "plan_execute", "context_managed"]


@pytest.mark.parametrize("name", ALL)
def test_text_only_replies_do_not_burn_the_call_budget(name, cfg, tmp_path):
    # pilot bug: providers ignore tool_choice="required"; a model that thinks it is done keeps
    # answering in text, and the harness used to nudge it until max_model_calls
    rec, fake = _run(name, cfg, tmp_path, FakeLLM(text_after=3))
    assert rec["status"] == "ok"
    # each agent (orchestrator, meta, each worker or executor) gets at most 2 idle replies:
    # plan_execute = 3 calls + 3 workers x 2 + orchestrator x 2 = 11, far below max_model_calls (80)
    assert rec["model_calls"] <= 11
    assert rec["evaluations"] < 7


@pytest.mark.parametrize("name", ["baseline_autoresearch", "plan_execute", "context_managed"])
def test_finish_stops_the_run(name, cfg, tmp_path):
    rec, fake = _run(name, cfg, tmp_path, FakeLLM(finish_after=4))
    assert rec["status"] == "ok" and rec["evaluations"] < 7
    assert len(rec["curve"]) == rec["evaluations"]   # curves still defined; carried forward when plotting


def _reason_requests(fake):
    return [m for _, m, names in fake.calls if not names]     # step 1 of each round: no tools offered


def test_baseline_autoresearch_is_append_only(cfg, tmp_path):
    _, fake = _run("baseline_autoresearch", cfg, tmp_path)
    reqs = _reason_requests(fake)
    for prev, cur in zip(reqs, reqs[1:]):
        assert cur[: len(prev) - 1] == prev[:-1]      # the history only grows (last message = this round's ask)
    assert len(reqs[-1]) > len(reqs[0])


def test_baseline_autoresearch_reason_then_act_then_record(cfg, tmp_path):
    rec, fake = _run("baseline_autoresearch", cfg, tmp_path)
    roles_tools = [(r, bool(n)) for r, _, n in fake.calls]
    assert roles_tools[:2] == [("agent", False), ("agent", True)]      # round 1: reason (no tools), then act
    act = fake.calls[1][1]
    assert act[-2] == {"role": "assistant", "content": "text reply #1"}   # the reasoning precedes the action
    records = [m["content"] for m in _reason_requests(fake)[-1] if m["content"].startswith("Round 1\n")
               or m["content"].startswith("Round 2\n")]
    assert records[0].startswith("Round 1\nReasoning: text reply #1\nTool call: evaluate_equation(c0*x + c1)"
                                 "\nResult: train R^2")
    assert records[1].startswith("Round 2\nReasoning: text reply #3\n")


def test_baseline_autoresearch_without_reasoning(cfg, tmp_path):
    cfg["experiment"]["harnesses"]["baseline_autoresearch"]["reasoning"] = False
    rec, fake = _run("baseline_autoresearch", cfg, tmp_path)
    assert rec["evaluations"] == 7
    assert all(names for _, _, names in fake.calls)            # every call offers tools: no reason step
    assert len(fake.calls) == 7                                 # one call per round
    last = fake.calls[-1][1]
    rec1 = next(m["content"] for m in last if m["content"].startswith("Round 1\n"))
    assert rec1.startswith("Round 1\nTool call: evaluate_equation(c0*x + c1)\nResult: ")
    assert "Reasoning" not in rec1


def test_plan_execute_follows_the_slide(cfg, tmp_path):
    rec, fake = _run("plan_execute", cfg, tmp_path)               # k = 3 workers, 7 evaluations
    roles = [r for r, _, _ in fake.calls]
    assert roles[:4] == ["orchestrator", "worker", "worker", "worker"]   # plan, then k workers
    # every worker: a fresh context (task + its direction) and exactly one equation
    workers = [m for r, m, _ in fake.calls if r == "worker"]
    assert len(workers) == rec["evaluations"] == 7
    assert all(len(m) == 2 and m[1]["content"].startswith("Research direction: ") for m in workers)
    # the orchestrator's conversation is the shared history: it grows with plans and results
    orch = [m for r, m, _ in fake.calls if r == "orchestrator"]
    assert len(orch) == 3 and len(orch[1]) > len(orch[0])          # rounds of 3, 3, 1 evaluations
    results = orch[1][-1]["content"]
    assert results.startswith("Results of round 1") and "equation: c0*x + c1" in results


def test_plan_execute_diverse_adds_one_instruction(cfg, tmp_path):
    from sr_harness.prompts.plan_execute import DIVERSITY
    cfg["experiment"]["harnesses"]["plan_execute"]["diverse"] = True
    _, fake = _run("plan_execute", cfg, tmp_path)
    system = [m for r, m, _ in fake.calls if r == "orchestrator"][0][0]["content"]
    assert DIVERSITY in system
    _, plain = _run("plan_execute", load_config(), tmp_path)
    assert DIVERSITY not in [m for r, m, _ in plain.calls if r == "orchestrator"][0][0]["content"]


def test_context_managed_follows_the_slide(cfg, tmp_path):
    rec, fake = _run("context_managed", cfg, tmp_path)           # 7 evaluations = 7 steps
    roles = [r for r, _, _ in fake.calls]
    # steps 1-2: meta, executor (no earlier step besides the last to select); then curator, meta, executor
    assert roles[:4] == ["meta", "executor"] * 2 and roles[4:7] == ["curator", "meta", "executor"]
    # the meta-reasoner and the executor start fresh every step: system prompt + one message
    meta = [m for r, m, _ in fake.calls if r == "meta"]
    executor = [m for r, m, _ in fake.calls if r == "executor"]
    assert len(meta) == len(executor) == rec["evaluations"] == 7
    assert all(len(m) == 2 for m in meta + executor)
    assert all(m[1]["content"] == "Work order: try sin terms in t" for m in executor)
    # m_t is the template: budget, best equations, last step, steps selected by the curator
    m7 = meta[-1][1]["content"]
    assert "Remaining budget: 1 evaluations" in m7 and "Best equations so far:" in m7
    assert "Last step:\n  Step 6:" in m7 and "selected by the curator:\n  Step 1:" in m7   # [1, 99] -> step 1
    # the curator sees every earlier step except the last, one line each
    curator = [m for r, m, _ in fake.calls if r == "curator"]
    assert curator[-1][1]["content"].count("Step ") == 5          # steps 1..5 before choosing step 7


def test_prompt_prefix_shared_across_roles(cfg, tmp_path):
    _, fake = _run("context_managed", cfg, tmp_path)
    systems = {m[0]["content"].split("# Your role")[0] for _, m, _ in fake.calls}
    assert len(systems) == 1                                      # identical task block for every role


def test_no_hidden_data_reaches_agents(cfg, tmp_path):
    task = toy_tasks()[0]
    fields = {f.name for f in dataclasses.fields(TaskView)}
    assert not fields & {"x_train_dist", "y_train_dist", "x_ood", "y_ood", "ground_truth"}
    _, fake = _run("plan_execute", cfg, tmp_path)
    text = " ".join(str(m) for _, ms, _ in fake.calls for m in ms)
    assert f"{task.y_train_dist[0]:.4g}" not in text and f"{task.y_ood[0]:.4g}" not in text
    assert task.ground_truth not in text
