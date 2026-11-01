"""Copy the runs the notebooks use from runs/ to recorded/, without the benchmark data.

LLM-SRBench is gated, so the published runs must not contain its data. Every model call's
prompt includes 30 training rows; the export drops the prompts and raw model replies from
the logs, keeping the first prompt of each run with its data rows replaced by a placeholder
(the reasoning-ablation notebook shows it). Everything the analysis reads is kept: results,
evaluations (equations, scores, reasoning), plans, work orders, curation, token counts.

    uv run python scripts/export_recorded.py
"""
import json
import re
import shutil
from pathlib import Path

from sr_harness import load_config
from sr_harness.analysis import multi_agent as MA
from sr_harness.analysis import plots
from sr_harness.benchmark.loader import load_llm_srbench

EXPERIMENTS = ["reasoning_ablation", "reasoning_ablation_16", "multi_agent", "context_managed_16"]
DROP = ("request", "response", "reasoning")         # fields of model_call events
ROWS = re.compile(r"(Training observations \(.*?\):\n[^\n]*\n)(.*?)(\n\n# The evaluate_equation tool)", re.S)
PLACEHOLDER = "  (data rows not included in the recorded runs: LLM-SRBench is gated; run live to see them)"


def redact(messages):
    return [{**m, "content": ROWS.sub(lambda g: g[1] + PLACEHOLDER + g[3], m["content"])}
            if isinstance(m.get("content"), str) else m for m in messages]


def export_log(src, dst):
    first = True
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            e = json.loads(line)
            if e["event"] == "model_call":
                request = e.get("request")
                e = {k: v for k, v in e.items() if k not in DROP}
                if first and request:
                    e["request"] = redact(request)
                    first = False
            fout.write(json.dumps(e) + "\n")


def main(src_root=Path("runs"), dst_root=Path("recorded")):
    for name in EXPERIMENTS:
        src, dst = src_root / name, dst_root / name
        shutil.rmtree(dst, ignore_errors=True)
        (dst / "logs").mkdir(parents=True)
        shutil.copytree(src / "results", dst / "results")
        for log in sorted((src / "logs").glob("*.jsonl")):
            export_log(log, dst / "logs" / log.name)
        print(f"{name}: {len(list((dst / 'results').glob('*.json')))} runs")
    # the worker-overlap table needs the training inputs (same predictions?): precompute it
    cfg = load_config()
    tasks = {t.task_id: t for t in load_llm_srbench(cfg["benchmark"]["subsets"], cfg["benchmark"]["data_dir"])}
    results = plots.load_results(dst_root / "multi_agent")
    MA.diversity_table(results, dst_root / "multi_agent", tasks).to_csv(dst_root / "multi_agent" / "overlap.csv",
                                                                       index=False)
    leaked = [p for p in dst_root.rglob("*") if p.is_file() and _has_rows(p)]
    assert not leaked, f"data rows left in {leaked[:3]}"


def _has_rows(path):
    text = path.read_text()
    return "Training observations" in text and PLACEHOLDER not in text


if __name__ == "__main__":
    main()
