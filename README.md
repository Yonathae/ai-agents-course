# AI Agents course — code

Code for the course *Agentic AI Systems*, Tel Aviv University, 2026.

Each folder is a self-contained project with its own README, requirements and notebook.

| Folder | Topic |
|---|---|
| [`harness-design-sql-agent/`](harness-design-sql-agent/) | Harness design: an SQL agent built from scratch, a benchmark and a paired A/B test |

## Getting started

```sh
git clone https://github.com/Yonathae/ai-agents-course.git
cd ai-agents-course/<folder>
```

Then follow the README in that folder.

## Model access

The notebooks call language models through [OpenRouter](https://openrouter.ai), which serves many models
behind one API. Each notebook can also replay recorded results, so you can run it without an API key.
To run live, create an OpenRouter API key, add a small amount of credit, and set it in your terminal:

```sh
export OPENROUTER_API_KEY=sk-or-...        # Windows (PowerShell): $env:OPENROUTER_API_KEY="sk-or-..."
```
