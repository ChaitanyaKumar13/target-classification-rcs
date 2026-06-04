from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def latest_run_dir(outputs_dir: Path) -> Path | None:
    runs = [p for p in outputs_dir.glob("cnn_run_*") if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_carrada_acc(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("carrada_test_acc"))
    except Exception:
        return None


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        cursor = out
        parts = k.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = v
    return out


def build_variants() -> list[dict]:
    # Ordered from conservative to aggressive.
    return [
        {
            "name": "no_sim_low_aug",
            "overrides": {
                "use_sim_data": False,
                "use_sim": False,
                "cnn.epochs": 60,
                "cnn.lr": 0.0006,
                "cnn.label_smoothing": 0.01,
                "augmentation.mask_prob": 0.10,
                "occlusion.prob": 0.15,
                "train_min_per_class": 500,
                "train_max_per_class": 1800,
            },
        },
        {
            "name": "no_sim_no_occlusion",
            "overrides": {
                "use_sim_data": False,
                "use_sim": False,
                "cnn.epochs": 75,
                "cnn.lr": 0.0005,
                "cnn.label_smoothing": 0.0,
                "augmentation.mask_prob": 0.05,
                "occlusion.enable": False,
                "train_min_per_class": 500,
                "train_max_per_class": 1600,
            },
        },
        {
            "name": "no_balance_high_epochs",
            "overrides": {
                "use_sim_data": False,
                "use_sim": False,
                "balance_train_only": False,
                "cnn.epochs": 90,
                "cnn.lr": 0.0004,
                "cnn.label_smoothing": 0.0,
                "augmentation.mask_prob": 0.08,
                "occlusion.prob": 0.10,
            },
        },
        {
            "name": "sim_small_weight",
            "overrides": {
                "use_sim_data": True,
                "use_sim": True,
                "sim_limit_per_class": 300,
                "cnn.epochs": 70,
                "cnn.lr": 0.0006,
                "cnn.label_smoothing": 0.01,
                "augmentation.mask_prob": 0.10,
                "occlusion.prob": 0.15,
                "train_min_per_class": 600,
                "train_max_per_class": 1800,
            },
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=str, default="src/config/experiment.yml")
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--python-bin", type=str, default="python")
    args = ap.parse_args()

    base_config = Path(args.base_config)
    outputs_dir = Path("outputs")
    sweep_dir = outputs_dir / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    base = load_yaml(base_config)
    variants = build_variants()
    history = []
    best_acc = -1.0
    best_run = None

    print(f"[INFO] Target Carrada accuracy: {args.target_acc:.4f}")
    print(f"[INFO] Max runs: {args.max_runs}")
    print(f"[INFO] Sweep dir: {sweep_dir}")

    run_count = 0
    while run_count < args.max_runs:
        variant = variants[run_count % len(variants)]
        cfg = apply_overrides(base, variant["overrides"])
        cfg["seed"] = int(base.get("seed", 42)) + run_count

        cfg_path = sweep_dir / f"config_{run_count+1:02d}_{variant['name']}.yml"
        save_yaml(cfg_path, cfg)

        before = set(p.name for p in outputs_dir.glob("cnn_run_*") if p.is_dir())
        cmd = [args.python_bin, "-B", "-m", "src.models.train_cnn", "--config", str(cfg_path)]
        print(f"\n[RUN {run_count+1}] variant={variant['name']} seed={cfg['seed']}")
        print("[CMD]", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        env["PYTHONIOENCODING"] = "utf-8"
        rc = subprocess.run(cmd, env=env).returncode
        after = set(p.name for p in outputs_dir.glob("cnn_run_*") if p.is_dir())
        new_runs = sorted(list(after - before))
        run_dir = outputs_dir / new_runs[-1] if new_runs else None

        acc = None
        if run_dir is not None and rc == 0:
            acc = read_carrada_acc(run_dir)

        record = {
            "run_index": run_count + 1,
            "variant": variant["name"],
            "seed": cfg["seed"],
            "return_code": rc,
            "run_dir": str(run_dir) if run_dir else None,
            "carrada_acc": acc,
            "config_path": str(cfg_path),
        }
        history.append(record)

        with open(sweep_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"[RESULT] return_code={rc} run_dir={record['run_dir']} carrada_acc={acc}")

        if acc is not None and acc > best_acc:
            best_acc = acc
            best_run = run_dir

        if acc is not None and acc >= args.target_acc:
            print(f"[SUCCESS] Target reached with accuracy={acc:.4f} at {run_dir}")
            break

        run_count += 1

    summary = {
        "target_acc": args.target_acc,
        "max_runs": args.max_runs,
        "best_acc": best_acc,
        "best_run": str(best_run) if best_run else None,
        "completed_runs": len(history),
        "history_path": str((sweep_dir / "history.json").resolve()),
    }
    with open(sweep_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SWEEP SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
