# AI Agents course — code

Code for the course *Agentic AI Systems*, Tel Aviv University.

## Lecture 2: an SQL agent

[`lecture-2/sql-agent.ipynb`](lecture-2/sql-agent.ipynb) builds an agent from scratch in plain Python and evaluates it:

1. query a small shop database with SQL by hand;
2. build an agent (a language model + tools + a loop) that answers questions by querying the database;
3. build a 20-task benchmark and compare the agent with a one-shot baseline in a paired A/B test.

## Quick start

```sh
git clone https://github.com/Yonathae/ai-agents-course.git
cd ai-agents-course/lecture-2
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab sql-agent.ipynb
```

Run the cells from the top.

- **No API key needed:** with `LIVE = False` (the default) the notebook replays recorded model results.
- **Run live with your own credits:** create an [OpenRouter](https://openrouter.ai) API key, set
  `export OPENROUTER_API_KEY=sk-or-...` in the terminal before starting Jupyter, and set `LIVE = True`
  in the first cell. The whole notebook costs about one cent.
