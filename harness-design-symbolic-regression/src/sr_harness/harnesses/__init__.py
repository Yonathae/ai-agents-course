from .base import Harness, HarnessResult, ResearchSession
from .context_managed import ContextManagedHarness
from .plan_execute import PlanExecuteHarness
from .baseline_autoresearch import BaselineAutoResearchHarness


def make_harness(name, cfg):
    """Build a harness from configs/experiment.yaml -> harnesses.<name>."""
    p = dict(cfg)
    if name.startswith("baseline_autoresearch"):     # H1 and its ablations (config entries)
        h = BaselineAutoResearchHarness(**p)
    elif name.startswith("plan_execute"):            # H2 and its variants (e.g. plan_execute_k4)
        h = PlanExecuteHarness(**p)
    elif name == "context_managed":
        h = ContextManagedHarness(**p)
    else:
        raise ValueError(f"unknown harness {name}")
    h.name = name                                    # results are labeled by the config entry
    return h
