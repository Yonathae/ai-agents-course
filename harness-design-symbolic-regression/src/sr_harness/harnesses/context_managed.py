"""H3 Context-Managed (MIRA-inspired; a simplified version, not MIRA's trained controller).

Each step t:
    m_t ~ M_cur( . | h_t)              curator: builds the context for this decision
    w_t ~ M_meta( . | m_t)             meta-reasoner: the next work order, fresh context
    f_t ~ M_exec( . | q, w_t)          executor: one equation, fresh context
    o_t = evaluate_equation(f_t)
    h_{t+1} = h_t + {(w_t, f_t, o_t)}  the history keeps everything (no model call)
One step costs exactly one evaluation, as in AutoResearch.

Memory and context are separate (Lecture 1): the history h keeps every step, but the
meta-reasoner never sees it. It sees m_t, rebuilt from scratch each step: a fixed template
(budget, best equations, last step) plus the earlier steps the curator selected. The
curator only selects steps; it does not rewrite them. No conversation grows step to step.
"""
from ..prompts.common import EVALUATE_TOOL, FINISH_TOOL, system_prompt
from ..prompts.context_managed import (ASSIGN_TOOL, CURATE_NUDGE, EXECUTOR, EXECUTOR_NUDGE, META, META_NUDGE,
                                       SELECT_TOOL, context_prompt, curate_prompt, curator_prompt, step_line,
                                       work_order_prompt)
from .base import Harness, force, request_action, user


class ContextManagedHarness(Harness):
    name = "context_managed"

    def __init__(self, selected_steps=5, best_equations=3):
        self.n_selected, self.n_best = selected_steps, best_equations

    def run(self, session):
        h = []                                       # history: one (work order, equation, result) per step
        t = 0
        while not session.exhausted() and session.can_call_model():
            t += 1
            # 1. curate: build the context m_t for this decision (a model call selects earlier steps)
            m = self.curate(session, h, t)

            # 2. meta-reasoner: a fresh context (task + m_t); one model call returns the work order w_t
            resp = request_action(session, "meta", [system_prompt(session.block, META), user(m)],
                                  [ASSIGN_TOOL, FINISH_TOOL], nudge=META_NUDGE)
            if any(c.name == "finish" for c in resp.tool_calls):
                session.log("stop", role="meta", reason="finish")
                break
            w = next((c.arguments.get("work_order") for c in resp.tool_calls if c.name == "assign_work"), None)
            if not w:                                # edge case: no work order (text or an unknown tool)
                break

            # 3. executor: a fresh context (task + w_t); one equation f_t, evaluated by the harness (o_t)
            f, o = run_executor(session, w, t)

            # 4. the history keeps everything (no model call)
            h.append((w, f, o))

        return session.result(self.name)

    def curate(self, session, h, t):
        """m_t: a fixed template filled from the history, plus the earlier steps the curator selects."""
        lines = [step_line(i, *step) for i, step in enumerate(h, 1)]
        selected = []
        if len(h) >= 2:                              # the curator chooses among steps before the last one
            resp = request_action(session, "curator",
                                  [system_prompt(session.block, curator_prompt(self.n_selected)),
                                   user(curate_prompt("\n".join(lines[:-1])))],
                                  [SELECT_TOOL], force("select_steps"), nudge=CURATE_NUDGE)
            call = next((c for c in resp.tool_calls if c.name == "select_steps"), None)
            numbers = call.arguments.get("steps", []) if call else []
            keep = [n for n in dict.fromkeys(numbers) if isinstance(n, int) and 1 <= n < len(h)]
            selected = [lines[n - 1] for n in sorted(keep[: self.n_selected])]
            session.log("curation", step=t, selected=keep[: self.n_selected])

        ok = sorted((s for s in h if s[2] is not None and s[2].ok), key=lambda s: -s[2].train_r2)
        best = [f"  {f}  (train R^2 = {o.train_r2:.4f})" for _, f, o in ok[: self.n_best]]
        return context_prompt(evaluations_left=session.evaluations_left(),
                              best="\n".join(best) or "  (none yet)",
                              last=lines[-1] if lines else "  (none yet)",
                              selected="\n".join(selected) or "  (none)")


def run_executor(session, work_order, t):
    """Step t's executor: a fresh context with the task and the work order; one equation."""
    session.log("executor_start", step=t, work_order=work_order)
    messages = [system_prompt(session.block, EXECUTOR), user(work_order_prompt(work_order))]
    resp = request_action(session, "executor", messages, [EVALUATE_TOOL], force("evaluate_equation"),
                          nudge=EXECUTOR_NUDGE)
    call = next((c for c in resp.tool_calls if c.name == "evaluate_equation"), None)
    if call is None:                                 # the executor never proposed an equation
        return "(none)", None
    r = session.evaluate(call.arguments.get("expression", ""), "executor")
    return r.expression, r
