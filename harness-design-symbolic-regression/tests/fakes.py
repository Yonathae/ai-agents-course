"""A scripted stand-in for LLMClient: same interface, no network."""
import itertools
import json

from sr_harness.llm import LLMResponse, ToolCall

EXPRESSIONS = ["c0*x + c1", "c0*x**2 + c1*t", "c0*x**2 + c1*sin(t)", "bad(x)", "c0*x**2 + c1*sin(c2*t)"]


class FakeLLM:
    def __init__(self, expressions=EXPRESSIONS, text_after=None, finish_after=None):
        """text_after=n: after n calls, reply with text only (ignoring tool_choice, as real
        providers sometimes do). finish_after=n: after n calls, call finish when offered."""
        self.expressions = itertools.cycle(expressions)
        self.calls = []            # (role, messages copy, tool names)
        self._ids = itertools.count()
        self.text_after, self.finish_after = text_after, finish_after

    def generate(self, messages, tools=None, tool_choice=None, role=None):
        names = [t["function"]["name"] for t in tools or []]
        self.calls.append((role, [dict(m) for m in messages], names))
        forced = tool_choice.get("function", {}).get("name") if isinstance(tool_choice, dict) else None
        if self.text_after is not None and len(self.calls) > self.text_after:
            return self._resp(messages, content="The fit is perfect; this is the final equation.")
        if self.finish_after is not None and len(self.calls) > self.finish_after and "finish" in names:
            forced = "finish"
        if not tools:
            return self._resp(messages, content=f"text reply #{len(self.calls)}")   # reasoning step, curator brief
        name = forced or ("evaluate_equation" if "evaluate_equation" in names else names[0])
        args = {
            "evaluate_equation": lambda: {"expression": next(self.expressions)},
            "submit_plan": lambda: {"directions": [d for d in
                                                       ("polynomial", "trigonometric", "exponential",
                                                        "rational")]},
            "select_steps": lambda: {"steps": [1, 99]},                      # 99: out of range, ignored
            "assign_work": lambda: {"work_order": "try sin terms in t"},
            "submit_rating": lambda: {"score": 4, "reasoning": "clear"},
            "finish": lambda: {"reason": "good enough"},
        }[name]()
        call = {"id": f"call_{next(self._ids)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        return self._resp(messages, tool_calls=[call], reasoning=f"hypothesis #{len(self.calls)}")

    @staticmethod
    def _resp(messages, content="", tool_calls=None, reasoning=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        n_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResponse(message=msg, content=content, reasoning=reasoning,
                           tool_calls=[ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]),
                                                c["function"]["arguments"]) for c in tool_calls or []],
                           input_tokens=n_in, output_tokens=20, cost=1e-6, model="fake")
