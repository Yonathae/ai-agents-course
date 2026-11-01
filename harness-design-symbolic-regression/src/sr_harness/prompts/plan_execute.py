"""H2 Plan-and-Execute: the orchestrator and the workers."""

# the ablation (diverse=True): one extra sentence in the orchestrator prompt
DIVERSITY = """Give every worker a different kind of direction (for example: polynomial,
trigonometric, exponential or logarithmic, rational), so that the team explores
different options in parallel."""


def orchestrator_prompt(k, extra_instructions=""):
    """The orchestrator's role (system prompt); extra_instructions: DIVERSITY or empty."""
    return f"""You lead a research team discovering the equation behind the data.
You do not write equations yourself. Each round you call submit_plan with {k} research
direction(s), one per worker. Each worker proposes one equation for its direction, and
the harness evaluates it on the training data. Use the results of earlier rounds to plan
the next one. If you are confident that no better equation can be found, call finish
instead of submit_plan.
{extra_instructions}""".rstrip()


START = "Plan the first round."

PLAN_NUDGE = "Call submit_plan, or finish if no better equation can be found."


def results_message(t, results, evaluations_left):
    """What the orchestrator learns from round t: each worker's direction, equation and result."""
    workers = "\n".join(f"[{j}] direction: {direction}\n    equation: {equation}\n    result: {feedback}"
                        for j, (direction, equation, feedback) in enumerate(results, 1))
    return f"Results of round {t} ({evaluations_left} evaluations left):\n{workers}\n\nPlan the next round."


PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the research directions for this round, one per worker.",
        "parameters": {
            "type": "object",
            "properties": {
                "directions": {
                    "type": "array",
                    "items": {"type": "string", "description": "what one worker should try, in one sentence"},
                },
            },
            "required": ["directions"],
        },
    },
}

# --- workers: a fresh context with the task and one direction; one equation ----------------------
WORKER = """You are a worker in a research team. You receive one research direction.
Propose one equation that follows it and evaluate it with evaluate_equation."""


def direction_prompt(direction):
    return f"Research direction: {direction}"


WORKER_NUDGE = "Call evaluate_equation with one equation for your direction."
