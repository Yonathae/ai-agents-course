"""Record live results to disk; replay them when the API is unavailable."""
import json
from dataclasses import asdict
from pathlib import Path

from .agent import Run

RECORDED = Path(__file__).resolve().parent.parent / "recorded"


def cached(name, fn, live):
    """live=True: call fn(), save its result as recorded/<name>.json.
    live=False: load the saved result instead."""
    path = RECORDED / f"{name}.json"
    if live:
        obj = fn()
        RECORDED.mkdir(exist_ok=True)
        if isinstance(obj, Run):
            obj.to_json(path)
        else:
            json.dump(obj, open(path, "w"), indent=1, default=_runs)
        return obj
    if not path.exists():
        raise FileNotFoundError(f"No recording {path.name}: run once with LIVE = True.")
    if name.startswith("run_"):
        return Run.from_json(path)
    return json.load(open(path))


def _runs(o):
    if isinstance(o, Run):
        return asdict(o)
    raise TypeError(type(o))
