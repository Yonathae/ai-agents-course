"""H2 Plan-and-Execute (multi-agent orchestration).

Each round t:
    P_t = {p_t1, ..., p_tk} ~ M_orch( . | q, h_t)      the orchestrator plans k directions
    f_tj ~ M( . | q, p_tj)                              worker j: one equation, fresh context
    o_tj = evaluate_equation(f_tj)                      the harness evaluates it
    h_{t+1} = h_t + {P_t, (f_tj, o_tj) for j = 1..k}    shared history; replan next round
A round costs exactly k evaluations, so a budget of B evaluations allows B/k rounds.

Memory: the orchestrator's conversation is the shared history h (plans and results).
A worker sees only the task and its direction. Workers are independent, so they could
run in parallel; here they run one after another (same tokens, same evaluations).

diverse=True adds one sentence to the orchestrator prompt: give every worker a different
kind of direction (the ablation for the task-distribution question).
"""
from ..prompts.common import EVALUATE_TOOL, FINISH_TOOL, system_prompt
from ..prompts.plan_execute import (DIVERSITY, PLAN_NUDGE, PLAN_TOOL, START, WORKER, WORKER_NUDGE,
                                    direction_prompt, orchestrator_prompt, results_message)
from .base import Harness, force, request_action, tool_result, user


class PlanExecuteHarness(Harness):
    name = "plan_execute"

    def __init__(self, workers=3, diverse=False):
        self.k, self.diverse = workers, diverse

    def run(self, session):
        role = orchestrator_prompt(self.k, DIVERSITY if self.diverse else "")
        h = [system_prompt(session.block, role), user(START)]          # the shared history
        t, finished = 0, False

        while not finished and not session.exhausted() and session.can_call_model():
            t += 1
            # 1. plan: one model call returns k research directions (nudge once if it replies in text)
            resp = request_action(session, "orchestrator", h, [PLAN_TOOL, FINISH_TOOL], nudge=PLAN_NUDGE)
            if not resp.tool_calls:
                break

            for call in resp.tool_calls:
                if call.name == "finish":
                    session.log("stop", role="orchestrator", reason=call.arguments.get("reason", "finish"))
                    h.append(tool_result(call, "Research finished."))
                    finished = True
                elif call.name == "submit_plan":
                    P = [p for p in call.arguments.get("directions", []) if isinstance(p, str) and p][: self.k]
                    session.log("plan", round=t, directions=P)
                    # 2. workers: one equation each, evaluated by the harness
                    results = [run_worker(session, p, t, j) for j, p in enumerate(P) if not session.exhausted()]
                    # 3. shared history: the plan, every worker's equation and result
                    h.append(tool_result(call, results_message(t, results, session.evaluations_left())))
                else:                                # edge case: the model called a tool that does not exist
                    h.append(tool_result(call, f"ERROR: unknown tool {call.name}"))

        return session.result(self.name)


def run_worker(session, direction, t, j):
    """Worker j of round t: a fresh context with the task and one direction; one equation."""
    session.log("worker_start", round=t, worker=j, direction=direction)
    messages = [system_prompt(session.block, WORKER), user(direction_prompt(direction))]
    resp = request_action(session, "worker", messages, [EVALUATE_TOOL], force("evaluate_equation"),
                          nudge=WORKER_NUDGE)
    call = next((c for c in resp.tool_calls if c.name == "evaluate_equation"), None)
    if call is None:                                 # the worker never proposed an equation
        return direction, "(none)", "no equation proposed"
    r = session.evaluate(call.arguments.get("expression", ""), "worker")
    return direction, r.expression, r.feedback().replace("\n", "; ")
