"""Load symbolic-regression tasks.

llm-srbench  LSR-Synth subsets of LLM-SRBench (Shojaee et al., 2025), the subsets
             with an extrapolation split. Hosted on Hugging Face (gated): accept the
             conditions at https://huggingface.co/datasets/nnheui/llm-srbench and
             log in (`hf auth login`). Format (checked against the file): one
             HDF5 file, group /lsr_synth/<subset>/<name> with arrays train
             (4000 rows), test (500, inside the training range) and ood_test
             (500, about 11% beyond it); column 0 is the target, columns 1.. the
             inputs, in the order of `symbols`.
             Our names: train -> train, the benchmark's test -> ID,
             the benchmark's ood_test -> OOD.
toy          two small synthetic tasks in the same format, for tests and
             offline dry runs.
"""
import numpy as np

from .tasks import SymbolicRegressionTask

REPO_ID = "nnheui/llm-srbench"
DATA_FILE = "lsr_bench_data.hdf5"

DOMAINS = {   # one-line context per LSR-Synth subset, shown to the agent
    "chem_react": "chemical reaction kinetics",
    "bio_pop_growth": "population growth in biology",
    "phys_osc": "a nonlinear oscillator in physics",
    "matsci": "material science",
}


def load_tasks(bench_cfg):
    if bench_cfg["source"] == "toy":
        return toy_tasks()
    tasks = load_llm_srbench(bench_cfg["subsets"], bench_cfg.get("data_dir"))
    return select_tasks(tasks, bench_cfg["n_tasks"], bench_cfg["task_seed"])


def load_llm_srbench(subsets, data_dir=None):
    import datasets
    import h5py
    from huggingface_hub import hf_hub_download

    datasets.disable_progress_bars()

    h5_path = hf_hub_download(REPO_ID, DATA_FILE, repo_type="dataset", cache_dir=data_dir)
    meta = datasets.load_dataset(REPO_ID, cache_dir=data_dir)
    tasks = []
    with h5py.File(h5_path, "r") as f:
        for subset in subsets:
            for e in meta[f"lsr_synth_{subset}"]:
                g = f[f"/lsr_synth/{subset}/{e['name']}"]
                train, train_dist = g["train"][...], g["test"][...]
                ood = g["ood_test"][...] if "ood_test" in g else None
                tasks.append(_make_task(f"{subset}/{e['name']}", subset, e["symbols"], e["symbol_descs"],
                                        train, train_dist, ood, e.get("expression")))
    return tasks


def select_tasks(tasks, n, seed):
    """n tasks, spread evenly over subsets, chosen with a fixed seed."""
    rng = np.random.default_rng(seed)
    by_domain = {}
    for t in tasks:
        by_domain.setdefault(t.domain, []).append(t)
    for ts in by_domain.values():
        rng.shuffle(ts)
    chosen, i = [], 0
    while len(chosen) < min(n, len(tasks)):
        for ts in by_domain.values():
            if i < len(ts) and len(chosen) < n:
                chosen.append(ts[i])
        i += 1
    return chosen


def _make_task(task_id, domain, symbols, descs, train, train_dist, ood, expression=None):
    split = lambda a: (None, None) if a is None else (a[:, 1:].astype(float), a[:, 0].astype(float))
    (x_tr, y_tr), (x_td, y_td), (x_od, y_od) = split(train), split(train_dist), split(ood)
    target, inputs = symbols[0], list(symbols[1:])
    description = (f"Domain: {DOMAINS.get(domain, domain)}.\n"
                   f"Discover an equation for {target} ({descs[0]}) as a function of "
                   + ", ".join(f"{s} ({d})" for s, d in zip(inputs, descs[1:])) + ".")
    return SymbolicRegressionTask(task_id, domain, description, target, descs[0], inputs, list(descs[1:]),
                                  x_tr, y_tr, x_td, y_td, x_od, y_od, expression)


def toy_tasks(seed=0):
    """Two synthetic tasks: train, ID (same range), OOD (beyond the range)."""
    rng = np.random.default_rng(seed)

    def sample(f, lo, hi, n, d):
        x = rng.uniform(lo, hi, size=(n, d))
        return np.column_stack([f(x), x])

    f1 = lambda x: 0.8 * x[:, 0] ** 2 - 1.5 * np.sin(x[:, 1])
    f2 = lambda x: 2.0 * x[:, 0] * np.exp(-0.5 * x[:, 0])
    return [
        _make_task("toy/quadratic_sine", "toy", ["y", "x", "t"], ["the response", "position", "time"],
                   sample(f1, -2, 2, 200, 2), sample(f1, -2, 2, 100, 2), sample(f1, 2, 4, 100, 2),
                   "0.8*x**2 - 1.5*sin(t)"),
        _make_task("toy/damped_growth", "toy", ["P", "t"], ["the population", "time"],
                   sample(f2, 0, 4, 200, 1), sample(f2, 0, 4, 100, 1), sample(f2, 4, 8, 100, 1),
                   "2*t*exp(-0.5*t)"),
    ]
