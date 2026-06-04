import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# -------- CONFIG --------
# If MODEL_PATH is None, script auto-selects best run by highest Carrada test accuracy.
MODEL_PATH = None
RUNS_DIR = Path("outputs")
CLASS_NAMES = ["car", "pedestrian", "cyclist", "drone", "truck", "tank"]
TARGET95_OP_PATH = Path("outputs/sixclass_target95_t075.json")

# Example: load one RD frame (.npy)
SAMPLE_RD = Path("matlab/04_train_with_sim_data/sim_data_npy/truck/truck_0035.npy")


def normalize_sample(x, eps=1e-6):
    x = x.astype(np.float32)
    m = x.mean()
    s = x.std()
    if s < eps:
        s = eps
    return (x - m) / (s + eps)


def find_best_model_path(runs_dir: Path) -> Path:
    best_acc = -1.0
    best_model = None

    for run_dir in sorted(runs_dir.glob("cnn_run_*")):
        metrics_path = run_dir / "metrics.json"
        model_path = run_dir / "best_model.keras"
        if not metrics_path.exists() or not model_path.exists():
            continue

        try:
            import json
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            acc = float(metrics.get("carrada_test_acc", -1.0))
        except Exception:
            continue

        if acc > best_acc:
            best_acc = acc
            best_model = model_path

    if best_model is None:
        raise FileNotFoundError(
            f"No valid model found under {runs_dir}. Expected cnn_run_*/best_model.keras + metrics.json"
        )

    print(f"Selected best run model: {best_model} (carrada_test_acc={best_acc:.4f})")
    return best_model


def read_target95_operating_point(path: Path) -> tuple[Path | None, float | None, float | None]:
    if not path.exists():
        return None, None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            op = json.load(f)
        model_path = Path(op.get("model_path")) if op.get("model_path") else None
        threshold = float(op.get("threshold")) if op.get("threshold") is not None else None
        temperature = float(op.get("temperature")) if op.get("temperature") is not None else None
        return model_path, threshold, temperature
    except Exception:
        return None, None, None


def apply_temperature(probs: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None or temperature <= 0:
        return probs
    p = probs.astype(np.float64) ** (1.0 / float(temperature))
    p = p / np.sum(p, axis=-1, keepdims=True)
    return p.astype(np.float32)


def main(
    model_path: Path | None = None,
    sample_path: Path | None = None,
    threshold: float | None = None,
    temperature: float | None = None,
):
    if sample_path is None:
        sample_path = SAMPLE_RD
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample RD file not found: {sample_path}")

    if model_path is None:
        model_path = MODEL_PATH

    op_model, op_threshold, op_temperature = read_target95_operating_point(TARGET95_OP_PATH)
    if model_path is None and op_model is not None:
        model_path = op_model
        print(f"Using target95 model from {TARGET95_OP_PATH}: {model_path}")
    if threshold is None and op_threshold is not None:
        threshold = op_threshold
        print(f"Using target95 threshold from {TARGET95_OP_PATH}: {threshold:.2f}")
    if temperature is None and op_temperature is not None:
        temperature = op_temperature
        print(f"Using target95 temperature from {TARGET95_OP_PATH}: {temperature:.2f}")

    if model_path is None:
        model_path = find_best_model_path(RUNS_DIR)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded")

    print(f"Loading RD sample: {sample_path}")
    rd = np.load(sample_path)

    if rd.ndim == 2:
        rd = rd[..., np.newaxis]  # (H,W,1)

    rd = normalize_sample(rd)
    rd = np.expand_dims(rd, axis=0)  # (1,H,W,1)

    print(f"Input shape: {rd.shape}")

    probs = model.predict(rd, verbose=0)[0]
    probs = apply_temperature(probs[np.newaxis, :], temperature=temperature)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    conf = float(np.max(probs))

    print("\nPrediction probabilities:")
    for c, p in zip(CLASS_NAMES, probs):
        print(f"  {c:12s}: {p:.4f}")

    print(f"\nFinal prediction: {pred_class.upper()} (confidence={conf:.4f})")
    if threshold is not None:
        decision = "ACCEPTED" if conf >= float(threshold) else "REJECTED"
        print(f"Decision at threshold {float(threshold):.2f}: {decision}")
    if temperature is not None:
        print(f"Applied temperature scaling: T={float(temperature):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Path to .keras model. If omitted, best run is auto-selected.")
    ap.add_argument("--sample", type=str, default=None, help="Path to .npy RD sample")
    ap.add_argument("--threshold", type=float, default=None, help="Optional rejection threshold")
    ap.add_argument("--temperature", type=float, default=None, help="Optional temperature for confidence calibration")
    args = ap.parse_args()

    main(
        model_path=Path(args.model) if args.model else None,
        sample_path=Path(args.sample) if args.sample else None,
        threshold=args.threshold,
        temperature=args.temperature,
    )
