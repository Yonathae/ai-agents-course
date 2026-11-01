"""External experiment logging: what the experimenter sees.

The logger is passive. It writes events to a JSONL file and never returns them
to a harness, so logging cannot leak information into an agent's context.
What an agent may remember is decided by its harness (see harnesses/), not here.
"""
import json
import time
from pathlib import Path

import numpy as np


class RunLogger:
    def __init__(self, path, run_info=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self.t0 = time.time()
        if run_info:
            self.log("run_start", **run_info)

    def log(self, event, **fields):
        rec = {"event": event, "t": round(time.time() - self.t0, 3), **fields}
        self._f.write(json.dumps(rec, default=_jsonable) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def read_log(path):
    return [json.loads(line) for line in open(path)]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
