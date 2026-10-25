# Harness design: an SQL agent

[`sql-agent.ipynb`](sql-agent.ipynb) builds an agent from scratch in plain Python and evaluates it:

1. query a small shop database with SQL by hand;
2. build an agent (a language model + tools + a loop) that answers questions by querying the database;
3. build a 20-task benchmark and compare the agent with a one-shot baseline in a paired A/B test.

## Quick start

From the repository root, with [uv](https://docs.astral.sh/uv/) (recommended):

```sh
cd harness-design-sql-agent
uv run --with-requirements requirements.txt jupyter lab sql-agent.ipynb
```

uv creates an isolated environment with the packages from `requirements.txt` (cached after the first run)
and starts Jupyter in it. Without uv, use pip (Python 3.9 or newer):

```sh
cd harness-design-sql-agent
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab sql-agent.ipynb
```

Run the cells from the top.

- **No API key needed:** with `LIVE = False` (the default) the notebook replays recorded model results
  from `recorded/`.
- **Run live with your own credits:** set `OPENROUTER_API_KEY` in the terminal before starting Jupyter
  (see the [main README](../README.md)) and set `LIVE = True` in the first cell.
  The whole notebook costs about one cent with the default model.

## Files

```
sql-agent.ipynb      the notebook
sql_agent/           the code: database, tools, agent loop, model calls, benchmark and statistics
recorded/            recorded model results, replayed when LIVE = False
requirements.txt     Python packages
```
