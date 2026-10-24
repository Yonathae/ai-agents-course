"""Two harnesses around the same model.

run_agent        iterative model-tool loop; ends when the model calls submit_answer
run_single_shot  one model call: schema in the prompt, one SQL query out

Both return a Run: the answer, the SQL whose result is scored, a readable
trace, and the metrics used in the evaluation part of the lecture.
"""
import json
import re
import time
from dataclasses import dataclass, field, asdict

from . import tools as T
from .llm import chat

SYSTEM_PROMPT = """You answer questions about a SQLite database.
You do not know the schema: use the tools to inspect it.
Explore with execute_sql. When done, call submit_answer with the query
that returns exactly the answer (no extra columns or rows)."""

NUDGE = "Call submit_answer with the query that returns exactly the answer."


@dataclass
class Metrics:
    model_calls: int = 0
    tool_calls: int = 0
    t_llm: float = 0.0       # seconds waiting for the model
    t_tools: float = 0.0     # seconds executing tools
    t_total: float = 0.0     # end to end
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0        # USD, as reported by OpenRouter

    @property
    def t_harness(self):
        return self.t_total - self.t_llm - self.t_tools

    def add_model(self, s):
        self.model_calls += 1
        self.t_llm += s["latency"]
        self.tokens_in += s["tokens_in"]
        self.tokens_out += s["tokens_out"]
        self.cost += s["cost"]


@dataclass
class Run:
    question: str
    answer: str = ""
    final_sql: str = None    # its result is what the benchmark compares
    status: str = "ok"       # ok | max_steps | tool_error | no_sql (single-shot)
    trace: list = field(default_factory=list)   # (speaker, text)
    metrics: Metrics = field(default_factory=Metrics)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        d["metrics"] = Metrics(**d["metrics"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Harness B: the iterative agent (the execution loop)
# ---------------------------------------------------------------------------

def run_agent(client, model, db_path, question, max_steps=10, recover=True):
    run = Run(question)                  # collects the answer, trace and metrics
    t_start = time.perf_counter()
    # the conversation so far: instructions + question; it grows every step
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    run.trace.append(("USER", question))

    for _ in range(max_steps):           # step limit: the loop cannot run forever
        # 1-2. send the conversation and the tool descriptions; get the reply
        msg, stats = chat(client, model, messages, tools=T.TOOL_SCHEMAS)
        run.metrics.add_model(stats)                     # latency, tokens, cost
        messages.append(msg.model_dump(exclude_none=True))   # keep the reply in the history

        # a reply without a tool call (rare): remind the model to submit
        if not msg.tool_calls:
            run.trace += [("MODEL", msg.content or ""), ("HARNESS", NUDGE)]
            messages.append({"role": "user", "content": NUDGE})
            continue

        # 3-4. execute each tool call and append its result as an observation
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments   # args: a JSON string
            obs = _execute(run, db_path, name, args)                   # the harness runs the tool
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
            if obs.startswith("ERROR"):
                # with recovery the model reads the error next step; without it we stop
                if not recover:
                    run.status = "tool_error"
            elif name == "submit_answer":
                # the submitted query's result is the agent's answer
                run.final_sql, run.answer = json.loads(args)["query"], obs

        # 5. stop after a successful submit_answer, or on an error without recovery
        if run.final_sql or run.status == "tool_error":
            break
    else:
        run.status = "max_steps"         # the step limit ran out before a submit

    run.metrics.t_total = time.perf_counter() - t_start   # end-to-end time
    return run


def _execute(run, db_path, name, args):
    """Run one tool call; record it in the trace and the metrics."""
    run.trace.append(("MODEL", f"{name}({_fmt_args(args)})"))
    t0 = time.perf_counter()
    obs = T.call_tool(db_path, name, args)
    run.metrics.t_tools += time.perf_counter() - t0
    run.metrics.tool_calls += 1
    run.trace.append(("TOOL", obs))
    return obs


class SQLAgent:
    """An agent = a model + a harness (tools and loop) around one database."""

    def __init__(self, client, model, db_path, max_steps=10, recover=True):
        self.client, self.model, self.db_path = client, model, db_path
        self.max_steps, self.recover = max_steps, recover
        self.tools = T.TOOL_SCHEMAS          # what the model is told about the tools

    def call_tool(self, name, **args):
        """Call a tool by name, exactly as the loop does when the model asks for it."""
        return T.call_tool(self.db_path, name, json.dumps(args))

    def run(self, question):
        """Answer a question with the execution loop (run_agent)."""
        return run_agent(self.client, self.model, self.db_path, question,
                         self.max_steps, self.recover)


# ---------------------------------------------------------------------------
# Harness A: single-shot text-to-SQL
# ---------------------------------------------------------------------------

SINGLE_SHOT_PROMPT = """SQLite schema:
{schema}

Write one SQLite query that answers the question. The query must return
exactly the answer (no extra columns or rows).
Reply with the query only, inside ```sql ... ```.

Question: {question}"""


def run_single_shot(client, model, db_path, question):
    run = Run(question)
    m = run.metrics
    t_start = time.perf_counter()
    prompt = SINGLE_SHOT_PROMPT.format(schema=T.schema_text(db_path),
                                       question=question)
    run.trace.append(("USER", question))
    msg, stats = chat(client, model, [{"role": "user", "content": prompt}])
    m.add_model(stats)
    sql = extract_sql(msg.content or "")
    run.trace.append(("MODEL", sql or msg.content))
    if sql is None:
        run.status = "no_sql"
    else:
        t0 = time.perf_counter()
        obs = T.execute_sql(db_path, sql)
        m.t_tools += time.perf_counter() - t0
        m.tool_calls += 1
        run.trace.append(("TOOL", obs))
        if obs.startswith("ERROR"):
            run.status = "tool_error"     # stop on error: no recovery
        else:
            run.final_sql, run.answer = sql, obs
    m.t_total = time.perf_counter() - t_start
    return run


def extract_sql(text):
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = (match.group(1) if match else text).strip()
    return sql or None


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_args(args_json):
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return args_json
    return ", ".join(repr(v) if k != "query" else "\n    " + v.strip()
                     for k, v in args.items())


def print_trace(run, max_lines=12):
    for who, text in run.trace:
        lines = str(text).splitlines() or [""]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
        print(f"{who}:")
        for line in lines:
            print("   ", line)
    print(f"[status: {run.status}]")


def print_metrics(run):
    m = run.metrics
    print(f"model calls      {m.model_calls}")
    print(f"tool calls       {m.tool_calls}")
    print(f"LLM latency      {m.t_llm:7.3f} s")
    print(f"tool latency     {m.t_tools * 1000:7.2f} ms")
    print(f"harness overhead {m.t_harness * 1000:7.2f} ms")
    print(f"end-to-end       {m.t_total:7.3f} s")
    print(f"tokens in / out  {m.tokens_in} / {m.tokens_out}")
    print(f"cost             ${m.cost:.6f}")
