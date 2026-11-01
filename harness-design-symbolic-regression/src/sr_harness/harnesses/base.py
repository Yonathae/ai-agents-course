"""What all harnesses share, and what they do not.

Shared (fairness): the model client, the evaluator, the budget, the logger, the
task block, and the final selection rule (return the evaluated equation with the
best training R^2). All of it goes through one ResearchSession per run.

Not shared (the architecture): how each harness organizes its memory and its
conversations. That lives in the harness modules.

ResearchSession keeps framework bookkeeping (counts, best-so-far) for the
experimenter. Harnesses must not read it to make decisions: anything an agent
knows must reach it through that harness's own messages or memory.
"""
from dataclasses import dataclass, field

from ..prompts.common import task_block


@dataclass
class HarnessResult:
    task_id: str
    harness: str
    best_expression: str | None
    best_params: dict
    best_train_r2: float | None
    evaluations: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost: float
    tokens_by_role: dict = field(default_factory=dict)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class ResearchSession:
    def __init__(self, task_view, evaluator, llm, budget, logger, prompt_cfg, max_params):
        self.task = task_view
        self.block = task_block(task_view, prompt_cfg["n_rows_shown"], max_params)
        self.evaluator, self.llm, self.budget, self.logger = evaluator, llm, budget, logger
        self.best = None                 # best EvaluationResult by training R^2
        self.input_tokens = self.output_tokens = self.cached_input_tokens = 0
        self.cost = 0.0
        self.tokens_by_role = {}

    # -- every model call -------------------------------------------------------
    def generate(self, role, messages, tools=None, tool_choice=None):
        resp = self.llm.generate(messages, tools=tools, tool_choice=tool_choice, role=role)
        self.budget.charge_model_call(resp.total_tokens)
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cached_input_tokens += resp.cached_input_tokens
        self.cost += resp.cost
        self.tokens_by_role[role] = self.tokens_by_role.get(role, 0) + resp.total_tokens
        return resp

    def can_call_model(self):
        return self.budget.model_calls < self.budget.max_model_calls

    # -- every equation evaluation ------------------------------------------------
    def evaluate(self, expression, role, reasoning=""):
        result = self.evaluator.evaluate(expression)
        self.budget.charge_evaluation()
        if result.ok and (self.best is None or result.train_r2 > self.best.train_r2):
            self.best = result
        # cumulative counters at this evaluation -> curves vs. evaluations and vs. tokens
        self.logger.log("evaluation", role=role, index=self.budget.evaluations, reasoning=reasoning,
                        expression=result.expression,
                        train_r2=result.train_r2, train_nmse=result.train_nmse, params=result.params,
                        n_nodes=result.n_nodes, error=result.error, tokens=self.budget.tokens,
                        model_calls=self.budget.model_calls, cost=self.cost)
        return result

    def exhausted(self):
        return self.budget.exhausted()

    def evaluations_left(self):
        return self.budget.remaining_evaluations()

    def log(self, event, **fields):
        self.logger.log(event, **fields)

    def result(self, harness_name):
        b = self.best
        return HarnessResult(self.task.task_id, harness_name, b and b.expression, b.params if b else {},
                             b and b.train_r2, self.budget.evaluations, self.budget.model_calls,
                             self.input_tokens, self.output_tokens, self.cached_input_tokens, self.cost,
                             dict(self.tokens_by_role))


class Harness:
    name = "base"

    def run(self, session: ResearchSession) -> HarnessResult:
        raise NotImplementedError


# -- one agent step ----------------------------------------------------------------

# Providers do not always enforce tool_choice="required": a model that considers itself done may
# answer in plain text instead of calling a tool. Reminding it forever burns the call and token
# budget (pilot 1: ~190k tokens per H1 run), so after this many text replies in a row we stop.
MAX_IDLE_REPLIES = 2


def request_action(session, role, messages, tools, tool_choice="required",
                   nudge="Call one of the available tools."):
    """Ask the model for its next action (a tool call); its reply is appended to `messages`.

    If it answers in text, append `nudge` (a reminder to call a tool) and ask again.
    Returns the response. If `resp.tool_calls` is still empty, the model answered in text
    MAX_IDLE_REPLIES times in a row: the caller should treat that agent as finished.
    """
    for attempt in range(MAX_IDLE_REPLIES):
        resp = session.generate(role, messages, tools, tool_choice)
        messages.append(resp.message)
        if resp.tool_calls:
            return resp
        if attempt + 1 < MAX_IDLE_REPLIES and session.can_call_model():
            messages.append(user(nudge))                 # text instead of an action: remind once
        else:
            break
    session.log("stop", role=role, reason="no tool call")
    return resp


# -- message helpers ---------------------------------------------------------------

def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def tool_result(call, content):
    return {"role": "tool", "tool_call_id": call.id, "content": content}


def force(tool_name):
    return {"type": "function", "function": {"name": tool_name}}
