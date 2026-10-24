"""OpenRouter through its OpenAI-compatible API, with timing and token counts."""
import os
import time

from openai import OpenAI

DEFAULT_MODEL = "openai/gpt-oss-20b"  # cheapest open-weight model with reliable tool calls


def make_client():
    """A client for OpenRouter: the standard OpenAI client pointed at another URL."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in the environment.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(client, model, messages, tools=None, temperature=0.0):
    """One model call. Returns (message, stats) where stats has latency,
    token counts and the cost OpenRouter reports (USD)."""
    # the request: which model, the conversation so far, and sampling settings
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  extra_body={"usage": {"include": True}})  # ask for cost
    if tools:
        # "required": every reply is a tool call, so the model cannot end the
        # episode with an empty or chatty message (it must call submit_answer)
        kwargs.update(tools=tools, tool_choice="required")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)      # one round trip to the model
    latency = time.perf_counter() - t0
    u = resp.usage                       # token counts (and cost) reported by the API
    stats = {
        "latency": latency,
        "tokens_in": getattr(u, "prompt_tokens", 0) or 0,
        "tokens_out": getattr(u, "completion_tokens", 0) or 0,
        "cost": (getattr(u, "model_extra", None) or {}).get("cost", 0.0) or 0.0,
    }
    return resp.choices[0].message, stats    # the reply: text and/or tool calls
