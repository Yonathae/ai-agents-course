"""Load the three YAML files in configs/ into one dict."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # harness-design-symbolic-regression/
CONFIG_DIR = ROOT / "configs"


def load_config(config_dir=CONFIG_DIR):
    config_dir = Path(config_dir)
    return {name: yaml.safe_load(open(config_dir / f"{name}.yaml"))
            for name in ("model", "benchmark", "experiment")}
