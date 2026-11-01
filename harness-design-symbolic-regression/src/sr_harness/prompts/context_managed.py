"""H3 Context-Managed: the curator, the meta-reasoner and the executor."""


def step_line(i, work_order, equation, result):
    """One step of the history as one line (what the curator reads)."""
    return f"  Step {i}: work order: {work_order[:200]} | equation: {equation} | {score(result)}"


def score(result):
    if result is None:
        return "no equation"
    if result.ok:
        return f"train R^2 = {result.train_r2:.4f}"
    return f"error: {result.error}"


# --- 1. curator: selects the earlier steps that matter for the next decision ---------------------
def curator_prompt(n):
    """The curator's role (system prompt): select at most n earlier steps."""
    return f"""You curate the research history of a team discovering the equation behind the data.
You see every earlier step, one per line. Select the steps (at most {n}) that matter most for
deciding what to investigate next: for example the best results, directions that failed,
and evidence about what has not been tried. Call select_steps with their numbers."""


def curate_prompt(history):
    return f"Research history:\n{history}\n\nSelect the relevant steps."


CURATE_NUDGE = "Call select_steps."

SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_steps",
        "description": "Select the earlier steps the research lead should see.",
        "parameters": {
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "integer"}}},
            "required": ["steps"],
        },
    },
}

# --- 2. meta-reasoner: sees only the curated context m_t, writes the next work order --------------
META = """You direct a research process discovering the equation behind the data.
You do not see the full history, only a curated view of it: the best equations so far, the
last step, and earlier steps selected by a curator. Choose the most useful next step and
write a work order for an executor, who starts fresh, sees only the task and your work
order, and proposes one equation. Call assign_work, or finish if no better equation can
be found."""


def context_prompt(evaluations_left, best, last, selected):
    """m_t: a fixed template, filled from the history."""
    return f"""Remaining budget: {evaluations_left} evaluations.

Best equations so far:
{best}

Last step:
{last}

Earlier steps selected by the curator:
{selected}

Write the next work order."""


META_NUDGE = "Call assign_work, or finish if no better equation can be found."

ASSIGN_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_work",
        "description": "Give the executor its work order for this step.",
        "parameters": {
            "type": "object",
            "properties": {"work_order": {"type": "string", "description": "precise instructions, "
                                          "including any equation to start from"}},
            "required": ["work_order"],
        },
    },
}

# --- 3. executor: a fresh context with the task and the work order; one equation -----------------
EXECUTOR = """You are an executor in a research team. You receive one work order.
Propose one equation that follows it and evaluate it with evaluate_equation."""


def work_order_prompt(work_order):
    return f"Work order: {work_order}"


EXECUTOR_NUDGE = "Call evaluate_equation with one equation for your work order."
