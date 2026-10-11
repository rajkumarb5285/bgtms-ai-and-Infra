"""
BGTMS Configuration Loader
"""

import os
from pathlib import Path
import yaml
from typing import Dict, Any


def load_yaml(filepath: str) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent.parent
        path = base_dir / filepath
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings() -> Dict[str, Any]:
    return load_yaml("configs/settings.yaml")


def get_classes() -> Dict[str, Any]:
    return load_yaml("configs/classes.yaml")
