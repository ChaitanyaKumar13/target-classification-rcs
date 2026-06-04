from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


DEFAULT_CLASSES = ["car", "pedestrian", "cyclist", "drone", "truck", "tank"]
RUNS_DIR = Path("outputs")
OP_FILES = [
    Path("outputs/sixclass_target95_t075.json"),
    Path("outputs/target95_operating_point.json"),
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        :root {
            --bg: #f2f4f3;
            --card: #ffffff;
            --ink: #102a43;
            --muted: #486581;
            --accent: #0b6e4f;
            --accent-2: #ef4444;
            --line: #cfd8dc;
        }
        html, body, [class*="css"], p, label, span, div {
            font-family: "Space Grotesk", "Manrope", sans-serif;
            color: var(--ink) !important;
        }
        .stApp {
            background:
                radial-gradient(1000px 500px at 0% 0%, #e3f7ef 0%, transparent 62%),
                radial-gradient(700px 350px at 100% 0%, #ffe8db 0%, transparent 58%),
                var(--bg);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 6px 20px rgba(16, 42, 67, 0.06);
        }
        .kpi-title { color: var(--muted); font-size: 0.85rem; }
        .kpi-value { color: var(--ink); font-size: 1.35rem; font-weight: 700; }
        .badge {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid var(--line);
            padding: 2px 10px;
            font-size: 0.75rem;
            color: var(--muted);
            background: #fff;
        }
        .result-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 6px 20px rgba(16, 42, 67, 0.06);
        }
        .result-title {
            margin: 0 0 8px 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink);
        }
        .result-row {
            margin: 4px 0;
            color: var(--ink);
        }
        .status-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #fff !important;
        }
        .status-accepted { background: #0b6e4f; }
        .status-rejected { background: #e15759; }
        .hero {
            background: linear-gradient(120deg, #0b6e4f, #175676);
            color: #ffffff;
            border-radius: 18px;
            padding: 20px;
            border: 1px solid #0a4f3a;
            box-shadow: 0 10px 26px rgba(11, 110, 79, 0.28);
        }
        h1, h2, h3, h4 {
            color: var(--ink) !important;
        }
        .hero h1, .hero p {
            color: #ffffff !important;
        }
        .hero h1 {
            font-size: clamp(2rem, 3.6vw, 3rem);
            line-height: 1.1;
            letter-spacing: 0.3px;
            margin: 0;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: 10px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.28);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: #e8f7ff !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: #334e68 !important;
            font-size: 0.9rem;
        }
        [data-testid="stTextInputRootElement"] input {
            background: #ffffff !important;
            color: #102a43 !important;
            border: 1px solid #bcccdc !important;
            border-radius: 10px !important;
        }
        [data-testid="stTextInputRootElement"] input::placeholder {
            color: #7b8794 !important;
        }
        [data-testid="stFileUploader"] {
            background: #ffffff !important;
            border: 1px solid #bcccdc !important;
            border-radius: 12px !important;
            padding: 8px;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {
            color: #486581 !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] div {
            color: #102a43 !important;
        }
        [data-testid="stMarkdownContainer"] p {
            color: #102a43 !important;
        }
        [data-testid="stButton"] button {
            background: linear-gradient(120deg, #ef4444, #f97316) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.1rem !important;
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.28) !important;
        }
        [data-testid="stButton"] button:hover {
            filter: brightness(1.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_float(v, default: float | None = None) -> float | None:
    try:
        return float(v)
    except Exception:
        return default


def format_percent(v: float | None) -> str:
    vv = 0.0 if v is None else float(v)
    return f"{vv * 100.0:.2f}%"


def normalize_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(x.mean())
    s = float(x.std())
    if s < eps:
        s = eps
    return (x - m) / (s + eps)


def load_metrics(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def gather_runs(runs_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(runs_dir.glob("cnn_run_*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = load_metrics(metrics_path)
        if not m:
            continue

        rej_main = ((m.get("rejection") or {}).get("results") or {}).get("main_threshold") or {}
        rows.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "carrada_acc": safe_float(m.get("carrada_test_acc")),
                "sim_acc": safe_float((m.get("sim_eval") or {}).get("sim_acc")),
                "coverage": safe_float(rej_main.get("coverage")),
                "acc_accepted": safe_float(rej_main.get("acc_accepted")),
                "rejected": rej_main.get("rejected"),
                "threshold": safe_float(rej_main.get("threshold")),
                "metrics": m,
            }
        )

    rows.sort(
        key=lambda r: (
            r["carrada_acc"] if r["carrada_acc"] is not None else -1.0,
            r["sim_acc"] if r["sim_acc"] is not None else -1.0,
        ),
        reverse=True,
    )
    return rows


def load_operating_point() -> dict | None:
    for p in OP_FILES:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            d["_path"] = str(p)
            return d
        except Exception:
            continue
    return None


def apply_temperature(probs: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None or temperature <= 0:
        return probs
    p = probs.astype(np.float64) ** (1.0 / float(temperature))
    p = p / np.sum(p, axis=-1, keepdims=True)
    return p.astype(np.float32)


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def choose_best_run(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    return runs[0]


def preprocess_uploaded_npy(uploaded_file, expected_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = expected_hw
    arr = np.load(uploaded_file)
    if arr.ndim == 2:
        arr2d = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr2d = arr[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"Expected .npy with shape (H,W) or (H,W,1), got {arr.shape}")

    # If shape differs, use image-like resize for practicality.
    if arr2d.shape != (h, w):
        pil = Image.fromarray(arr2d)
        pil = pil.resize((w, h))
        arr2d = np.array(pil, dtype=np.float32)

    arr2d = normalize_sample(arr2d)
    model_input = arr2d[..., np.newaxis]
    return arr2d, model_input


def plot_probability_bar(class_names: list[str], probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#0b6e4f" if i == int(np.argmax(probs)) else "#94a3b8" for i in range(len(probs))]
    ax.bar(class_names, probs, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    ax.tick_params(axis="x", rotation=35)
    return fig


def plot_input_image(img2d: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(img2d, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    return fig


def main():
    st.set_page_config(page_title="Target Classification Frontend", page_icon="RCS", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <span class="hero-kicker">Research Demo</span>
            <h1>Target Classification Using RCS</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    runs = gather_runs(RUNS_DIR)
    if not runs:
        st.error("No runs found under outputs/cnn_run_*. Train a model first.")
        return

    selected = choose_best_run(runs)
    if selected is None:
        st.error("Could not determine best run.")
        return

    op = load_operating_point()
    metrics = selected["metrics"]
    run_dir = Path(selected["run_dir"])
    class_names = metrics.get("class_names", DEFAULT_CLASSES)
    default_model = run_dir / "best_model.keras"
    op_threshold = None
    op_temp = None
    op_source = None

    if op:
        op_model = op.get("model_path")
        if op_model:
            cand = Path(op_model)
            if cand.exists():
                default_model = cand
        op_threshold = safe_float(op.get("threshold"))
        op_temp = safe_float(op.get("temperature"))
        op_source = op.get("_path")

    if not default_model.exists():
        st.error(f"Best run found, but model file missing: {default_model}")
        return

    occlusion_cfg = metrics.get("occlusion", {}) or {}
    rejection_cfg = metrics.get("rejection", {}) or {}
    rej_main = ((rejection_cfg.get("results") or {}).get("main_threshold") or {})
    rejection_threshold = safe_float(rejection_cfg.get("threshold"), 0.75) or 0.75
    if op_threshold is not None:
        rejection_threshold = op_threshold
    raw_acc = safe_float(selected.get("carrada_acc"), 0.0) or 0.0
    accepted_acc = safe_float(rej_main.get("acc_accepted"), 0.0) or 0.0
    coverage = safe_float(rej_main.get("coverage"), 0.0) or 0.0
    if op and op.get("acc_accepted") is not None:
        accepted_acc = safe_float(op.get("acc_accepted"), accepted_acc) or accepted_acc
    if op and op.get("coverage") is not None:
        coverage = safe_float(op.get("coverage"), coverage) or coverage
    data_shape = metrics.get("data_shape", [256, 64, 1])
    expected_h = int(data_shape[0]) if len(data_shape) >= 1 else 256
    expected_w = int(data_shape[1]) if len(data_shape) >= 2 else 64

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f"<div class='card'><div class='kpi-title'>Model Run</div><div class='kpi-value'>{Path(default_model).parent.name}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='card'><div class='kpi-title'>Test Accuracy</div><div class='kpi-value'>{format_percent(raw_acc)}</div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='card'><div class='kpi-title'>Accepted Accuracy</div><div class='kpi-value'>{format_percent(accepted_acc)}</div></div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='card'><div class='kpi-title'>Prediction Coverage</div><div class='kpi-value'>{format_percent(coverage)}</div></div>",
        unsafe_allow_html=True,
    )
    c5.markdown(
        f"<div class='card'><div class='kpi-title'>Occlusion / Rejection Threshold</div><div class='kpi-value'>{'On' if occlusion_cfg.get('enable', False) else 'Off'} / {rejection_threshold:.2f}</div></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Classify Target")
    st.write("Upload RD `.npy` data for inference.")
    target_text = st.text_input("Target Name / Context", placeholder="e.g., suspected target in frame")
    uploaded_npy = st.file_uploader("Upload RD Map (.npy)", type=["npy"])
    infer_clicked = st.button("Identify Target", type="primary")

    st.caption(f"Model input is resized to {expected_h}x{expected_w}.")

    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

    if infer_clicked:
        if uploaded_npy is None:
            st.error("Please upload a .npy RD file first.")
            return
        try:
            img2d, model_input = preprocess_uploaded_npy(uploaded_npy, (expected_h, expected_w))
            x = np.expand_dims(model_input, axis=0)

            model = load_model_cached(str(default_model.resolve()))
            probs = model.predict(x, verbose=0)[0]
            probs = apply_temperature(probs[np.newaxis, :], op_temp)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            pmax = float(np.max(probs))
            rejected = pmax < rejection_threshold
            top_idx = np.argsort(probs)[::-1][:3]
            top3 = [
                (
                    class_names[i] if i < len(class_names) else f"class_{i}",
                    float(probs[i]),
                )
                for i in top_idx
            ]
            calibration_method = (
                f"Temperature Scaling (T={op_temp:.2f})" if op_temp is not None else "None"
            )
            st.session_state["prediction_result"] = {
                "target_text": target_text.strip(),
                "pred_class": pred_class,
                "confidence": pmax,
                "rejected": rejected,
                "threshold": rejection_threshold,
                "calibration_method": calibration_method,
                "img2d": img2d,
                "probs": probs,
                "class_names": list(class_names),
                "top3": top3,
            }
        except Exception as e:
            st.error(f"Inference error: {e}")

    result = st.session_state.get("prediction_result")
    if result:
        status_text = "Rejected" if result["rejected"] else "Accepted"
        status_class = "status-rejected" if result["rejected"] else "status-accepted"
        interpretation_strength = "high" if result["confidence"] >= 0.9 else "moderate"
        interpretation = (
            f"The uploaded RD map was classified as {result['pred_class'].title()} "
            f"with {interpretation_strength} confidence and {status_text.lower()} under "
            f"the current rejection threshold."
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Prediction Result</div>
                <div class="result-row"><strong>Predicted Target:</strong> {result['pred_class'].title()}</div>
                <div class="result-row"><strong>Confidence Score:</strong> {result['confidence']:.4f}</div>
                <div class="result-row"><strong>Decision Status:</strong> <span class="status-badge {status_class}">{status_text}</span></div>
                <div class="result-row"><strong>Rejection Threshold:</strong> {result['threshold']:.2f}</div>
                <div class="result-row"><strong>Calibration Method:</strong> {result['calibration_method']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["target_text"]:
            st.caption(f"Target context: {result['target_text']}")
        st.caption(f"Input Type: RD Map (.npy)")
        st.write(interpretation)
        st.pyplot(plot_input_image(result["img2d"], "Preprocessed Range-Doppler Map"))
        st.pyplot(plot_probability_bar(result["class_names"], result["probs"]))
        st.markdown("**Top-3 Probabilities**")
        for name, score in result["top3"]:
            st.write(f"{name.title()}: {score:.4f}")


if __name__ == "__main__":
    main()
