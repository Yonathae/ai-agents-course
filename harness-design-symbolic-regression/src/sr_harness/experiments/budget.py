"""Computational budget, shared by all roles inside one run.

Two budgets are compared in the lecture:
  equation evaluations  - each call to evaluate_equation counts, valid or not
  inference tokens      - input + output tokens of every model call, all roles

max_model_calls is a safety stop only (a model that never calls the tool).
"""
from dataclasses import dataclass


@dataclass
class Budget:
    max_evaluations: int
    max_model_calls: int = 80
    max_tokens: int | None = None
    evaluations: int = 0
    model_calls: int = 0
    tokens: int = 0

    def remaining_evaluations(self):
        return max(0, self.max_evaluations - self.evaluations)

    def exhausted(self):
        return (self.evaluations >= self.max_evaluations
                or self.model_calls >= self.max_model_calls
                or (self.max_tokens is not None and self.tokens >= self.max_tokens))

    def charge_model_call(self, total_tokens):
        self.model_calls += 1
        self.tokens += total_tokens

    def charge_evaluation(self):
        self.evaluations += 1
