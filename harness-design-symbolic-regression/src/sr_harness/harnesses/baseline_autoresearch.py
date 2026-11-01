"""H1 AutoResearch (the ReAct pattern): one agent with the full history.

Each round t:
    reasoning_t ~ M( . | q, h_t)                  1. reason about the next tool call (no tools offered)
    f_t ~ M( . | q, h_t, reasoning_t)             2. act: one equation
    o_t = evaluate_equation(f_t)                  3. observe: its fit on the training data
    h_{t+1} = h_t + {reasoning_t, f_t, o_t}       the history keeps every round; it is resent every call
One round costs one evaluation. reasoning=False skips step 1 (the ablation in reasoning_ablation.ipynb).
"""
import re

from ..prompts.common import EVALUATE_TOOL, FINISH_TOOL, system_prompt
from ..prompts.baseline_autoresearch import (ACTION_PROMPT, NUDGE, WITH_REASONING, WITHOUT_REASONING, agent_prompt,
                                             reason_message, round_message, round_record)
from .base import Harness, assistant, request_action, user


class BaselineAutoResearchHarness(Harness):
    name = "baseline_autoresearch"

    def __init__(self, reasoning=True):
        self.reasoning = reasoning                   # True: reason, then act; False: act only

    def run(self, session):
        role = agent_prompt(WITH_REASONING if self.reasoning else WITHOUT_REASONING)
        h = [system_prompt(session.block, role)]     # the history: grows by one record per round
        finished, t = False, 0

        while not finished and not session.exhausted() and session.can_call_model():
            t += 1
            if self.reasoning:
                # 1. reason: ask for reasoning about the next tool call (no tools offered)
                reason_prompt = h + [user(reason_message(t))]
                reasoning = clean_reasoning(session.generate("agent", reason_prompt).content)
                action_prompt = reason_prompt + [assistant(reasoning), user(ACTION_PROMPT)]
            else:
                reasoning = None
                action_prompt = h + [user(round_message(t))]

            # 2. act: ask for the tool call (that follows the reasoning); nudge once if it replies in text
            resp = request_action(session, "agent", action_prompt, [EVALUATE_TOOL, FINISH_TOOL], nudge=NUDGE)
            if not resp.tool_calls:
                break

            for call in resp.tool_calls:             # usually exactly one
                if call.name == "finish":
                    session.log("stop", role="agent", reason=call.arguments.get("reason", "finish"))
                    content, finished = "Research finished.", True
                elif session.exhausted():
                    content = "ERROR: the evaluation budget is used up."
                elif call.name == "evaluate_equation":
                    r = session.evaluate(call.arguments.get("expression", ""), "agent", reasoning or "")
                    content = r.feedback() + f"\n({session.evaluations_left()} evaluations left)"
                else:                                # edge case: the model called a tool that does not exist
                    content = f"ERROR: unknown tool {call.name}"
                # 3. observe: keep a record of round t (reasoning, tool call, result)
                h.append(user(round_record(t, reasoning, call, content)))

        return session.result(self.name)


# Housekeeping: models sometimes start their reasoning with a heading ("**Reasoning for next step**",
# "Round 2 / Reasoning:"); drop it, and cap the length so the history cannot grow without bound.
MAX_REASONING_CHARS = 1500
_NEXT = r"for\s+(?:the\s+)?next\s+(?:tool\s+call|step)"
_LABEL = re.compile(r"^\s*(round\s*\d+[.:]?\s*)?"
                    rf"(\**\s*(?:(?:reasoning|rationale)(?:\s+{_NEXT})?|{_NEXT})\s*\**\s*:?\s*\**)?", re.I)


def clean_reasoning(text):
    return _LABEL.sub("", (text or "").strip(), count=1).strip()[:MAX_REASONING_CHARS]
