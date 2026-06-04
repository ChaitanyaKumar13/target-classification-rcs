from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.data_prep_6class import load_joint_6class_dataset


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(name: str, model, X_test, y_test, X_c_test, y_c_test, X_s_test, y_s_test):
    y_pred = model.predict(X_test)
    out = {
        "joint_test_acc": float(accuracy_score(y_test, y_pred)),
    }

    y_pred_c = model.predict(X_c_test)
    out["carrada_test_acc"] = float(accuracy_score(y_c_test, y_pred_c))

    if len(X_s_test) > 0:
        y_pred_s = model.predict(X_s_test)
        out["sim_test_acc"] = float(accuracy_score(y_s_test, y_pred_s))
    else:
        out["sim_test_acc"] = None

    print(
        f"[{name}] joint={out['joint_test_acc']:.4f} "
        f"carrada={out['carrada_test_acc']:.4f} "
        f"sim={out['sim_test_acc'] if out['sim_test_acc'] is not None else '-'}",
        flush=True,
    )
    return out


def main(config_path: str):
    cfg = load_yaml(config_path)
    data = load_joint_6class_dataset(cfg)

    X_train = data["X_train"].reshape(len(data["X_train"]), -1)
    y_train = data["y_train"].astype(int)
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)
    y_test = data["y_test"].astype(int)
    X_c_test = data["X_carrada_test"].reshape(len(data["X_carrada_test"]), -1)
    y_c_test = data["y_carrada_test"].astype(int)
    X_s_test = data["X_sim_test"].reshape(len(data["X_sim_test"]), -1)
    y_s_test = data["y_sim_test"].astype(int)

    out_dir = Path(cfg.get("paths", {}).get("out_dir", "outputs"))
    run_dir = out_dir / f"sklearn6_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pca_dim = int(cfg.get("sklearn6", {}).get("pca_components", 128))

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_dim, random_state=int(cfg.get("seed", 42)))),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, dual=False)),
        ]
    )
    svm.fit(X_train, y_train)
    svm_metrics = evaluate_model("LinearSVM", svm, X_test, y_test, X_c_test, y_c_test, X_s_test, y_s_test)

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_dim, random_state=int(cfg.get("seed", 42)))),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=350,
                    random_state=int(cfg.get("seed", 42)),
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model("RandomForest", rf, X_test, y_test, X_c_test, y_c_test, X_s_test, y_s_test)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": data["class_names"],
                "joint_test_size": int(len(y_test)),
                "carrada_test_size": int(len(y_c_test)),
                "sim_test_size": int(len(y_s_test)),
                "svm": svm_metrics,
                "random_forest": rf_metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved sklearn metrics to: {run_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/experiment_sixclass_goal95.yml")
    args = ap.parse_args()
    main(args.config)
