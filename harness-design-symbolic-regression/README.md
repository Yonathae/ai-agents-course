# Harness design II: research harnesses for symbolic regression

Three ways to organize iterative research around the same model, compared on the LSR-Synth tasks of
[LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench) (Shojaee et al., 2025). Each task gives
training data; the agent proposes equations, and the tool `evaluate_equation` fits their constants and returns the
fit. Every harness uses the same model, tool, budget and logging; only the loop differs.

| Harness | Slides | Code | One step |
|---|---|---|---|
| **H1 AutoResearch** | 12–14 | `harnesses/baseline_autoresearch.py` | one agent reasons, proposes one equation, sees the fit; the full history is resent every step |
| **H2 Plan-and-Execute** | 15–17 | `harnesses/plan_execute.py` | an orchestrator plans *k* directions; *k* workers propose one equation each |
| **H3 Context-Managed** | 18–20 | `harnesses/context_managed.py` | a curator selects earlier steps; a meta-reasoner writes a work order; a fresh executor proposes one equation |

Each harness file starts with the slide's pseudo-code and follows it step by step; its prompts are in
`prompts/<harness>.py`.

## Notebooks

| Notebook | Question |
|---|---|
| `reasoning_ablation.ipynb` | H1: how much does the reasoning step help? |
| `multi_agent.ipynb` | H2: do *k* workers converge *k* times faster? Do they explore different options? |
| `context_managed.ipynb` | H3: does curation beat the full history as runs get longer? |

Each notebook replays the recorded runs in `recorded/` (`LIVE = False`; no API key or dataset access needed) or
runs the experiment (`LIVE = True`). The recorded runs contain no benchmark data: LLM-SRBench is gated, so the
prompts' data rows are left out (`scripts/export_recorded.py`).

## Quick start

```sh
cd harness-design-symbolic-regression
uv run --extra dev jupyter lab                 # open a notebook; LIVE = False replays the recorded runs
uv run --extra dev pytest -q                   # offline tests (a scripted fake model; no API key)
```

To run live: set `OPENROUTER_API_KEY` (see `.env.example`). LLM-SRBench is gated: accept its conditions at
<https://huggingface.co/datasets/nnheui/llm-srbench> and log in with `uvx hf auth login`. Then, for example:

```sh
uv run python scripts/run_experiment.py --harness baseline_autoresearch --n-tasks 8 --seeds 0 --max-evaluations 5
```

Model: `openai/gpt-oss-20b` via OpenRouter (`configs/model.yaml`). A run costs about $0.001–0.005.

## Metrics

Each equation is scored on three splits:
- **train**: the data the agent searches on (it sees train R²);
- **ID** (in distribution): new points from the training range (hidden);
- **OOD**: points beyond the training range, about 11% further, mostly later in time (hidden).

Scores: **R²** = 1 − MSE / variance of the target within the split; **MSE** in the target's units, averaged over tasks
in log space (target scales differ by up to 16 orders of magnitude). At each time step a run's answer is its best
equation so far by train R².

## Layout

```
configs/            model.yaml, benchmark.yaml, experiment.yaml (harness variants)
src/sr_harness/
  harnesses/        base.py (shared: session, budget, one agent step), one file per harness
  prompts/          common.py (task block, tools), one file per harness
  benchmark/        tasks.py, loader.py (LLM-SRBench), evaluator.py (the evaluate_equation tool)
  evaluation/       metrics.py, holdout.py (hidden scores: ID, OOD)
  experiments/      runner.py (runs and per-step curves), budget.py, logging.py
  analysis/         plots.py, paired.py, multi_agent.py (H2), history.py (H3)
  llm.py            the one OpenRouter client used by every role
scripts/            run_experiment.py, compare.py, rescore.py, export_recorded.py (runs/ -> recorded/)
tests/              offline tests with a scripted fake model
recorded/           the runs the notebooks replay (results and logs, without the benchmark data)
*.ipynb             the three notebooks
```

Shared by every harness (`harnesses/base.py`): one `ResearchSession` per run counts every model call and every
evaluation against the budget and logs them; `request_action` asks the model for a tool call and nudges it once if it
answers in text. Harnesses decide only from their own messages, never from the log.
