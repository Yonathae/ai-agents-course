"""The task block shared by every role in every harness.

Prefix caching: each system prompt is  task block + role instructions.  The task
block is byte-identical across roles and calls within a run, so providers that
cache prompt prefixes can reuse it. Keep anything that changes (budget left,
findings) out of the system prompt and in later messages.
"""
import numpy as np

from ..benchmark.evaluator import FUNCTIONS


def task_block(view, n_rows_shown=30, max_params=8):
    """The task: description, variables, a sample of the training data, and how to call evaluate_equation."""
    names = list(view.variable_names)
    x = names[0]                                     # used in the example equation
    variables = "\n".join(f"  {n}: {d}" for n, d in zip(names, view.variable_descriptions))
    idx = np.linspace(0, len(view.y_train) - 1, min(n_rows_shown, len(view.y_train))).astype(int)
    header = "  ".join(f"{h:>11s}" for h in names + [view.target_name])
    rows = "\n".join("  ".join(f"{v:11.4g}" for v in list(view.x_train[i]) + [view.y_train[i]]) for i in idx)
    return f"""# Symbolic regression task
{view.description}

Target: {view.target_name} ({view.target_description})
Input variables:
{variables}

Training observations ({len(idx)} of {len(view.y_train)} rows; the evaluator uses all of them):
{header}
{rows}

# The evaluate_equation tool
Write an expression for {view.target_name} in Python syntax using the input variables, numbers,
free constants c0, c1, ... (fitted to the training data by least squares, at most {max_params}),
+ - * / **, and the functions {", ".join(FUNCTIONS)}, and the constants pi and e.
Example: c0*{x}**2 + c1*sin(c2*{x})
The tool returns the training R^2 and NMSE with the fitted constants, or an error.
Each call uses one evaluation from a limited budget."""


def system_prompt(block, role_instructions):
    """The system message of every model call: the task block, then the role."""
    return {"role": "system", "content": f"{block}\n\n# Your role\n{role_instructions}"}


FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Stop the research: call this only when you are confident no better equation can be found. "
                       "The best evaluated equation is returned automatically.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []},
    },
}

EVALUATE_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_equation",
        "description": "Fit the free constants of a candidate equation and return its training R^2 and NMSE.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "right-hand side, e.g. c0*x + c1"}},
            "required": ["expression"],
        },
    },
}
