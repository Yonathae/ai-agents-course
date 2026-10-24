# AI Agents course

Material for the course *Agentic AI Systems*, Tel Aviv University, 2026: lecture slides, notebooks and code.

The course is organized in modules: how AI systems are evaluated (benchmarks), what turns a language model
into an agent (the harness), AI safety, and post-training. Material is added here as the course progresses.

## Course outline

| Module | Lecture | Material |
|---|---|---|
| 1. Benchmarks | 1. A Historical Overview of AI | [slides](lectures/Lecture1_A_Historical_Overview_of_AI.pdf) |
| | 2. Foundations and Practice of AI Evaluation | [slides](lectures/Lecture2_Foundations_and_Practice_of_AI_Evaluation.pdf) |
| | 3. The Evolution of AI Benchmarking | [slides](lectures/Lecture3_The_Evolution_of_AI_Benchmarking.pdf) |
| 2. Harness design | 4. Foundations of Agent Harnesses | [slides](lectures/Lecture4_Foundations_of_Agent_Harnesses.pdf) · [notebook and code](harness-design-sql-agent/) |
| 3. AI safety | upcoming | |
| 4. Post-training | upcoming | |

## Repository layout

```
lectures/                   lecture slides (PDF)
harness-design-sql-agent/   lecture 4: an SQL agent built from scratch, a benchmark and a paired A/B test
```

Each code folder is a self-contained project with its own README, requirements and notebook.

## Running the notebooks

```sh
git clone https://github.com/Yonathae/ai-agents-course.git
cd ai-agents-course/<folder>
```

Then follow the README in that folder.

The notebooks call language models through [OpenRouter](https://openrouter.ai), which serves many models
behind one API. Each notebook can also replay recorded results, so you can run it without an API key.
To run live, create an OpenRouter API key, add a small amount of credit, and set it in your terminal:

```sh
export OPENROUTER_API_KEY=sk-or-...        # Windows (PowerShell): $env:OPENROUTER_API_KEY="sk-or-..."
```
