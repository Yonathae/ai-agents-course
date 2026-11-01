"""The common agent-facing tool: evaluate_equation(expression) -> EvaluationResult.

Identical for every harness. Steps:
  1. parse      the expression with Python's `ast` and accept only an allowlist
                (numbers, variables, constants c0..ck, + - * / **, math functions);
                model-generated code is never executed as Python
  2. fit        the free constants c0..ck by least squares on the training data
                (same seeded multi-start procedure for everyone)
  3. score      R^2 and NMSE on the training data only

The evaluator is built from a TaskView, so hidden held-out data is not even reachable.
Hidden scoring of a final equation uses `predict` (evaluation/metrics.py).
"""
import ast
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
}
CONSTANTS = {"pi": np.pi, "e": np.e}
PARAM = re.compile(r"^c(\d+)$")       # free constants are named c0, c1, ...
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


@dataclass
class EvaluationResult:
    expression: str
    train_r2: float | None = None
    train_nmse: float | None = None
    params: dict = field(default_factory=dict)    # fitted constants {"c0": 1.23, ...}
    n_nodes: int | None = None                    # expression size (a complexity measure)
    error: str | None = None

    @property
    def ok(self):
        return self.error is None

    def feedback(self):
        """The text the agent sees."""
        if not self.ok:
            return f"ERROR: {self.error}"
        consts = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items()) or "none"
        return (f"train R^2 = {self.train_r2:.6f}   train NMSE = {self.train_nmse:.3e}\n"
                f"fitted constants: {consts}")


class ExpressionError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------

def parse_expression(expression, variable_names, max_params=8, max_nodes=120):
    """Validate the expression; return (compiled code, sorted parameter names, node count)."""
    text = expression.strip().replace("^", "**")
    if "=" in text:                            # "y = ..." -> keep the right-hand side
        text = text.split("=")[-1].strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"syntax error: {e.msg}") from None

    params, n_nodes = set(), 0
    for node in ast.walk(tree):
        n_nodes += 1
        if isinstance(node, (ast.Expression, ast.Load) + _BINOPS + _UNARY):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS) or node.keywords:
                raise ExpressionError(f"unsupported function call; allowed: {', '.join(FUNCTIONS)}")
            continue
        if isinstance(node, ast.Name):
            if node.id in variable_names or node.id in CONSTANTS or node.id in FUNCTIONS:
                continue
            if PARAM.match(node.id):
                params.add(node.id)
                continue
            raise ExpressionError(f"unknown name '{node.id}'; variables are {', '.join(variable_names)}, "
                                  f"free constants are c0, c1, ...")
        raise ExpressionError(f"unsupported syntax: {type(node).__name__}")

    if n_nodes > max_nodes:
        raise ExpressionError(f"expression too large ({n_nodes} nodes, max {max_nodes})")
    if len(params) > max_params:
        raise ExpressionError(f"too many free constants ({len(params)}, max {max_params})")
    tree = _FloatConstants().visit(tree)       # 9**9**9 on floats overflows instead of hanging
    code = compile(ast.fix_missing_locations(tree), "<expression>", "eval")
    return code, sorted(params, key=lambda p: int(p[1:])), n_nodes


class _FloatConstants(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(float(node.value)), node)


def predict(code, variable_names, params, X):
    """Evaluate a parsed expression on inputs X (n, d) with constants `params` (dict)."""
    env = {"__builtins__": {}, **FUNCTIONS, **CONSTANTS, **params}
    env.update({name: X[:, j] for j, name in enumerate(variable_names)})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = eval(code, env)                # safe: the AST was checked against the allowlist
        except (OverflowError, ZeroDivisionError, ValueError, TypeError) as e:
            raise ExpressionError(f"numerical error: {e}") from None
    return np.broadcast_to(np.asarray(y, dtype=float), (X.shape[0],)).copy()


# ---------------------------------------------------------------------------
# 2-3. Fitting and scoring
# ---------------------------------------------------------------------------

def r2_score(y, y_pred):
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def nmse(y, y_pred):
    return float(np.mean((y - y_pred) ** 2) / np.var(y))


class Evaluator:
    """evaluate_equation for one task. Deterministic given the config."""

    def __init__(self, task_view, cfg):
        self.names = list(task_view.variable_names)
        self.X, self.y = task_view.x_train, task_view.y_train
        self.cfg = cfg

    def evaluate(self, expression) -> EvaluationResult:
        res = EvaluationResult(expression=str(expression))
        try:
            code, pnames, res.n_nodes = parse_expression(
                res.expression, self.names, self.cfg["max_params"], self.cfg["max_expression_nodes"])
            res.params = self._fit(code, pnames)
            y_pred = predict(code, self.names, res.params, self.X)
            bad = ~np.isfinite(y_pred)
            if bad.any():
                raise ExpressionError(f"non-finite predictions on {int(bad.sum())} of {len(y_pred)} "
                                      f"training points (overflow, log or sqrt of a negative, or 1/0)")
            res.train_r2, res.train_nmse = r2_score(self.y, y_pred), nmse(self.y, y_pred)
        except ExpressionError as e:
            res.error, res.params = str(e), {}
        return res

    def _fit(self, code, pnames):
        if not pnames:
            return {}

        def residuals(theta):
            r = predict(code, self.names, dict(zip(pnames, theta)), self.X) - self.y
            return np.where(np.isfinite(r), r, 1e6)      # keep the optimizer away from invalid regions

        rng = np.random.default_rng(self.cfg["fit_seed"])
        starts = [np.ones(len(pnames))] + [rng.normal(0, 2, len(pnames))
                                           for _ in range(self.cfg["n_restarts"] - 1)]
        best = None
        for x0 in starts:
            try:
                sol = least_squares(residuals, x0, max_nfev=self.cfg["max_nfev"])
            except (ValueError, ExpressionError):
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            raise ExpressionError("constant fitting failed")
        return {p: float(v) for p, v in zip(pnames, best.x)}
