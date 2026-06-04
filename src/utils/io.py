from pathlib import Path
import yaml

def ensure_dir(p: str | Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
