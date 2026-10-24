# AI Agents course

Material for the course *Agentic AI Systems*, Tel Aviv University, 2026.

## Lectures

| # | Lecture | Slides |
|---|---|---|
| 1 | A Historical Overview of AI | [PDF](lectures/Lecture1_A_Historical_Overview_of_AI.pdf) |
| 2 | Foundations and Practice of AI Evaluation | [PDF](lectures/Lecture2_Foundations_and_Practice_of_AI_Evaluation.pdf) |
| 3 | The Evolution of AI Benchmarking | [PDF](lectures/Lecture3_The_Evolution_of_AI_Benchmarking.pdf) |
| 4 | Foundations of Agent Harnesses | [PDF](lectures/Lecture4_Foundations_of_Agent_Harnesses.pdf) · code: [`harness-design-sql-agent/`](harness-design-sql-agent/) |

## Code

Each code folder is a self-contained project with its own README, requirements and notebook.

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
