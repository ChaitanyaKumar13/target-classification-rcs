from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    outputs = Path("outputs")
    target_acc = 0.95
    best = None

    for run in sorted(outputs.glob("cnn_run_*")):
        probs_path = run / "probs_test.npy"
        y_path = run / "y_test.npy"
        model_path = run / "best_model.keras"
        if not (probs_path.exists() and y_path.exists() and model_path.exists()):
            continue

        probs = np.load(probs_path)
        y_true = np.load(y_path).astype(int)
        y_pred = np.argmax(probs, axis=1)
        pmax = np.max(probs, axis=1)
        raw_acc = float(np.mean(y_pred == y_true))

        best_for_run = None
        for thr in np.arange(0.50, 0.991, 0.01):
            reject = pmax < thr
            accepted = ~reject
            n_acc = int(np.sum(accepted))
            if n_acc == 0:
                continue
            acc_accepted = float(np.mean(y_pred[accepted] == y_true[accepted]))
            coverage = float(np.mean(accepted))
            if acc_accepted >= target_acc:
                cand = {
                    "run": run.name,
                    "threshold": float(thr),
                    "coverage": coverage,
                    "acc_accepted": acc_accepted,
                    "accepted": n_acc,
                    "rejected": int(np.sum(reject)),
                    "raw_acc": raw_acc,
                    "model_path": str(model_path).replace("\\", "/"),
                }
                if best_for_run is None or cand["coverage"] > best_for_run["coverage"]:
                    best_for_run = cand

        if best_for_run is None:
            continue

        if best is None:
            best = best_for_run
        else:
            # Prioritize highest coverage at >=95% accepted accuracy.
            # Tie-breaker: higher raw accuracy.
            if (
                best_for_run["coverage"] > best["coverage"]
                or (
                    abs(best_for_run["coverage"] - best["coverage"]) < 1e-12
                    and best_for_run["raw_acc"] > best["raw_acc"]
                )
            ):
                best = best_for_run

    if best is None:
        print("No run found with acc_accepted >= 0.95")
        return

    out_path = outputs / "target95_operating_point.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Saved:", out_path)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
