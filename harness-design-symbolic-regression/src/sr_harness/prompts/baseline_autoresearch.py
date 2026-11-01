"""H1 Baseline AutoResearch: one agent, one continuing history of rounds."""

# how a round works: with the reasoning step (two calls) or without it (one call)
WITH_REASONING = """Each round has two steps: first you reason about your next tool call (what earlier
rounds showed and which hypothesis to test next), then you make that tool call.
Earlier rounds are kept in the conversation. Prefer simple equations that make sense
for the domain."""

WITHOUT_REASONING = """In each round, make one tool call: evaluate one candidate equation, using what the
earlier rounds showed. Earlier rounds are kept in the conversation. Prefer simple
equations that make sense for the domain."""


def agent_prompt(round_instructions):
    """The agent's role (system prompt)."""
    return f"""You are a scientist discovering the equation behind the data.
{round_instructions} When no better equation can be found, call finish."""


def reason_message(t):
    """Step 1 of round t: ask for the reasoning first (no tools offered)."""
    return f"""Round {t}. First, give your reasoning for the next tool call: what do the
earlier rounds show, which tool will you call (evaluate_equation with which equation,
or finish), and why? Answer in 1-3 sentences. Do not call a tool yet."""


# step 2 of round t: ask for the tool call that follows the reasoning
ACTION_PROMPT = "Now make the tool call your reasoning describes."


def round_message(t):
    """The single step of round t when there is no reasoning step."""
    return f"Round {t}. Make your next tool call."


def round_record(t, reasoning, call, result):
    """One round of the history, as the model reads it in later rounds (reasoning=None: no reasoning step)."""
    reasoning_line = "" if reasoning is None else f"Reasoning: {reasoning}\n"
    args = ", ".join(str(v) for v in call.arguments.values())
    return f"Round {t}\n{reasoning_line}Tool call: {call.name}({args})\nResult: {result}"


NUDGE = "Call evaluate_equation with your next candidate equation."
