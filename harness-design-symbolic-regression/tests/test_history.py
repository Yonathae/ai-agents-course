from fakes import FakeLLM
from sr_harness import load_config
from sr_harness.analysis import history
from sr_harness.benchmark import toy_tasks
from sr_harness.experiments.runner import run_one


class LoggingFake(FakeLLM):
    """Logs every call like the real LLMClient does (the fake alone does not write model_call events)."""

    def __init__(self, logger, **kw):
        super().__init__(**kw)
        self.logger = logger

    def generate(self, messages, tools=None, tool_choice=None, role=None):
        resp = super().generate(messages, tools, tool_choice, role)
        self.logger.log("model_call", role=role, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
        return resp


def test_calls_by_step_and_repeats(tmp_path):
    cfg = load_config()
    cfg["experiment"]["budget"]["max_evaluations"] = 6
    task = toy_tasks()[0]
    # the fake cycles through 3 equations, so evaluations 4-6 repeat evaluations 1-3
    make = lambda logger, seed: LoggingFake(logger, expressions=["c0*x + c1", "c0*x**2 + c1*t", "c0*sin(t) + c1"])
    rec = run_one(task, "context_managed", 0, cfg, make, tmp_path)

    calls = history.calls_by_step(tmp_path, [rec])
    assert set(calls.role) == {"curator", "meta", "executor"}
    assert calls.step.min() == 1 and calls.step.max() == 6
    assert set(calls[calls.step == 1].role) == {"meta", "executor"}       # nothing to curate at step 1

    rep = history.repeats(tmp_path, [rec], {task.task_id: task})
    assert rep.loc[0, "evaluations"] == 6 and rep.loc[0, "repeats"] == 3
