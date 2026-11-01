"""The one model client used by every harness and every role.

OpenRouter serves many models behind the OpenAI chat-completions API, so the
standard OpenAI client works with a different base URL. Every call returns an
LLMResponse with the message and all resource counts, and is written to the
experiment log. Harnesses never talk to OpenRouter directly.
"""
import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict          # parsed JSON; {} if the model sent invalid JSON
    raw_arguments: str


@dataclass
class LLMResponse:
    message: dict            # assistant message, ready to append to a conversation
    content: str | None
    reasoning: str | None    # only what the API exposes; hidden reasoning is not available
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0        # USD, as reported by OpenRouter
    latency: float = 0.0     # seconds
    model: str = ""
    provider: str | None = None
    request_id: str | None = None

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class LLMClient:
    def __init__(self, model_cfg, logger=None, seed=None):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY (see .env.example).")
        self.cfg = model_cfg
        self.model = model_cfg["model"]
        self.seed = seed
        self.logger = logger
        self.client = OpenAI(base_url=model_cfg["base_url"], api_key=key)

    def generate(self, messages, tools=None, tool_choice=None, role=None):
        """One model call. `role` (agent, planner, executor, ...) is only a log tag."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
            extra_body={
                "usage": {"include": True},                      # cost and cache counts
                "provider": self.cfg.get("provider", {}),
            },
        )
        if self.cfg.get("reasoning_effort"):                     # thinking models only; null for others
            kwargs["extra_body"]["reasoning"] = {"effort": self.cfg["reasoning_effort"]}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")

        t0 = time.perf_counter()
        resp = self._create_with_retries(kwargs)
        out = parse_response(resp, latency=time.perf_counter() - t0, model=self.model)
        if self.logger:
            self.logger.log("model_call", role=role, request=messages, tools=[t["function"]["name"] for t in tools or []],
                            response=out.message, reasoning=out.reasoning, **usage_fields(out))
        return out

    def _create_with_retries(self, kwargs):
        for attempt in range(self.cfg.get("max_retries", 3)):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:  # rate limits, provider hiccups
                if attempt == self.cfg.get("max_retries", 3) - 1:
                    raise
                if self.logger:
                    self.logger.log("model_retry", error=repr(e), attempt=attempt)
                time.sleep(2 ** attempt)


def parse_response(resp, latency, model):
    msg = resp.choices[0].message
    extra = msg.model_extra or {}
    u = resp.usage
    u_extra = (getattr(u, "model_extra", None) or {}) if u else {}
    details = getattr(u, "prompt_tokens_details", None)
    tool_calls = [ToolCall(c.id, c.function.name, _json_or_empty(c.function.arguments), c.function.arguments)
                  for c in msg.tool_calls or []]
    # Keep only content and tool calls in the conversation. Reasoning goes to the log, not back
    # into the prompt, so the conversation prefix stays stable (prefix caching) and comparable.
    message = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        message["tool_calls"] = [c.model_dump() for c in msg.tool_calls]
    return LLMResponse(
        message=message,
        content=msg.content,
        reasoning=extra.get("reasoning"),
        tool_calls=tool_calls,
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        cache_write_tokens=((getattr(details, "model_extra", None) or {}).get("cache_write_tokens", 0) or 0)
        if details else 0,
        cost=u_extra.get("cost", 0.0) or 0.0,
        latency=latency,
        model=getattr(resp, "model", model),
        provider=(resp.model_extra or {}).get("provider"),
        request_id=getattr(resp, "id", None),
    )


def usage_fields(r: LLMResponse):
    return dict(input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                cached_input_tokens=r.cached_input_tokens, cache_write_tokens=r.cache_write_tokens,
                cost=r.cost, latency=r.latency, model=r.model, provider=r.provider, request_id=r.request_id)


def _json_or_empty(s):
    try:
        out = json.loads(s or "{}")
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}
