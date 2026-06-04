# src/models/train_cnn.py
# ==========================================================
# WINDOWS CPU CRASH FIX (CRITICAL):
# ✅ Completely removes tf.data pipeline (which was silently crashing on some Windows CPU setups)
# ✅ Uses Keras Sequence (pure Python/Numpy batch generator) instead
# ✅ Keeps Carrada + MATLAB merge + train-only balancing
# ✅ Keeps augmentation (now done in NumPy safely)
#
# NEW (Research Add-ons):
# ✅ Occlusion phenomenon augmentation (range/doppler attenuation bands)
# ✅ Rejection Threshold (Unknown/Rejected) evaluation:
#    - coverage, accepted accuracy
#    - sweep thresholds 0.50..0.90
#
# PATCHES INCLUDED (All 4):
# 1) ✅ Keras warning fix: RadarSequence calls super().__init__(**kwargs)
# 2) ✅ YAML compat: supports use_sim_data OR use_sim
# 3) ✅ Honors occlusion.mode: "train" / "train+val" / "train+val+test"
# 4) ✅ Adds SIM-only evaluation split (defence targets) + reports SIM test metrics
#
# NEW (IEEE FIGURE EXPORT ADD-ON):
# ✅ Saves X_test.npy, y_test.npy, y_pred.npy, history.npy into each run_dir
#    so generate_figures.py can load and generate graphs without retraining
# ==========================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from src.data.loaders import load_carrada_dataset, balance_xy


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tf_runtime(cfg: dict):
    tf_cfg = cfg.get("tensorflow", {}) if isinstance(cfg.get("tensorflow", {}), dict) else {}
    intra = int(tf_cfg.get("intra_op_threads", 2))
    inter = int(tf_cfg.get("inter_op_threads", 2))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        print(f"🧠 TF threads: intra_op={intra}, inter_op={inter}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not set TF threading options: {e}", flush=True)

    # disable XLA
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass


def safe_bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return default


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = y.astype(int).reshape(-1)
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def normalize_per_sample(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = X.astype(np.float32, copy=False)

    if X.ndim != 4:
        raise RuntimeError(f"Unexpected X shape for normalize_per_sample: {X.shape}")

    # (N,1,H,W)
    if X.shape[1] == 1:
        for i in range(X.shape[0]):
            xi = X[i, 0]
            m = float(xi.mean())
            s = float(xi.std())
            if s < eps:
                s = eps
            X[i, 0] = (xi - m) / (s + eps)
        return X

    # (N,H,W,1)
    if X.shape[-1] == 1:
        for i in range(X.shape[0]):
            xi = X[i, :, :, 0]
            m = float(xi.mean())
            s = float(xi.std())
            if s < eps:
                s = eps
            X[i, :, :, 0] = (xi - m) / (s + eps)
        return X

    raise RuntimeError(f"Unexpected X shape for normalize_per_sample: {X.shape}")


def class_counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    y = y.astype(int)
    counts = np.bincount(y, minlength=len(class_names))
    return {class_names[i]: int(counts[i]) for i in range(len(class_names))}


def print_class_counts(tag: str, y: np.ndarray, class_names: list[str]) -> None:
    counts = class_counts(y, class_names)
    pretty = ", ".join([f"{k}={v}" for k, v in counts.items()])
    print(f"📊 {tag} class counts: {pretty}", flush=True)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def confusion_matrix_with_reject(y_true: np.ndarray, y_pred_with_reject: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Rows: true classes [0..n_classes-1]
    Cols: predicted classes [0..n_classes-1] plus last col = reject
    """
    cm = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred_with_reject.astype(int)):
        if p == -1:
            cm[t, n_classes] += 1
        else:
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t, p] += 1
    return cm


def classification_report_named(cm: np.ndarray, class_names: list[str]) -> dict:
    n_classes = cm.shape[0]
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = (correct / total) if total > 0 else 0.0

    report: dict = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report[class_names[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[c, :].sum()),
        }

    report["overall"] = {"accuracy": float(acc), "total": total}
    return report


def predicted_distribution(y_pred: np.ndarray, class_names: list[str]) -> dict[str, int]:
    y_pred = y_pred.astype(int)
    counts = np.bincount(y_pred, minlength=len(class_names))
    return {class_names[i]: int(counts[i]) for i in range(len(class_names))}


def reject_counts_by_true_class(y_true: np.ndarray, reject_mask: np.ndarray, class_names: list[str]) -> dict:
    out = {}
    y_true = y_true.astype(int)
    for i, name in enumerate(class_names):
        out[name] = int(np.sum(reject_mask & (y_true == i)))
    out["total_rejected"] = int(np.sum(reject_mask))
    return out


# -------------------------
# Stratified split (NumPy)
# -------------------------
def train_val_test_split_np(X, y, test_size=0.2, val_size=0.1, seed=42):
    rng = np.random.default_rng(seed)
    y = y.astype(int)

    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    test_idx_parts, rem_parts = [], []
    for c in classes:
        ix = idx_by_class[c]
        n_c = max(1, int(round(len(ix) * test_size)))
        test_idx_parts.append(ix[:n_c])
        rem_parts.append(ix[n_c:])

    test_idx = np.concatenate(test_idx_parts)
    rem_idx = np.concatenate(rem_parts)
    rng.shuffle(test_idx)
    rng.shuffle(rem_idx)

    n_val = max(1, int(round(len(rem_idx) * val_size))) if val_size > 0 else 0
    val_idx = rem_idx[:n_val] if n_val > 0 else np.array([], dtype=int)
    train_idx = rem_idx[n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


# =========================
# MATLAB synthetic RD loader
# =========================
def load_matlab_synthetic_rd(sim_root, class_names, input_shape, limit_per_class=None, seed=42, strict_shape=True, verbose=True):
    sim_root = Path(sim_root)
    rng = np.random.default_rng(seed)

    sim_classes = ["drone", "truck", "tank"]
    H, W, C = input_shape

    X_list, y_list = [], []
    per_class_counts = {}

    for cls in sim_classes:
        cls_dir = sim_root / cls
        files = sorted(cls_dir.glob("*.npy")) if cls_dir.exists() else []
        if len(files) == 0:
            if verbose:
                print(f"⚠️ No .npy found for {cls} in {cls_dir}", flush=True)
            continue

        if limit_per_class and len(files) > limit_per_class:
            idx = rng.permutation(len(files))[:limit_per_class]
            files = [files[i] for i in idx]

        cls_idx = class_names.index(cls)
        kept, skipped = 0, 0

        for fp in files:
            rd = np.load(fp)
            if rd.ndim == 2:
                rd = rd[..., np.newaxis]
            rd = rd.astype(np.float32, copy=False)

            if strict_shape and rd.shape != (H, W, C):
                skipped += 1
                continue

            X_list.append(rd)
            y_list.append(cls_idx)
            kept += 1

        per_class_counts[cls] = kept
        if verbose:
            print(f"✅ MATLAB loaded {kept} samples for '{cls}' (skipped shape={skipped})", flush=True)

    if len(X_list) == 0:
        raise RuntimeError(f"❌ No MATLAB synthetic RD samples found under: {sim_root}")

    X_sim = np.stack(X_list, axis=0)
    y_sim = np.array(y_list, dtype=np.int64)

    if verbose:
        print(f"🧪 MATLAB synthetic total: {len(X_sim)} samples", flush=True)
        print(f"🧪 MATLAB per-class: {per_class_counts}", flush=True)

    return X_sim, y_sim


# -------------------------
# NumPy augmentation (SAFE) + Occlusion
# -------------------------
def apply_numpy_occlusion(
    x: np.ndarray,
    rng: np.random.Generator,
    prob: float,
    range_band: tuple[int, int],
    doppler_band: tuple[int, int],
    attenuation: tuple[float, float],
):
    """
    x: (H,W,1)
    Occlusion = attenuated band on either Range (rows) OR Doppler (cols).
    """
    if rng.random() > float(prob):
        return x

    H, W, _ = x.shape
    which = int(rng.integers(0, 2))  # 0=range, 1=doppler
    att = float(rng.uniform(float(attenuation[0]), float(attenuation[1])))

    if which == 0:
        bw_min, bw_max = int(range_band[0]), int(range_band[1])
        bw_max = max(bw_min + 1, min(bw_max, H))
        bw = int(rng.integers(bw_min, bw_max + 1))
        start = int(rng.integers(0, max(1, H - bw + 1)))
        x[start:start + bw, :, :] *= att
    else:
        bw_min, bw_max = int(doppler_band[0]), int(doppler_band[1])
        bw_max = max(bw_min + 1, min(bw_max, W))
        bw = int(rng.integers(bw_min, bw_max + 1))
        start = int(rng.integers(0, max(1, W - bw + 1)))
        x[:, start:start + bw, :] *= att

    return x


def apply_numpy_aug(
    x: np.ndarray,
    rng: np.random.Generator,
    use_masking_aug: bool,
    mask_prob: float,
    range_mask_min: int,
    range_mask_max: int,
    doppler_mask_min: int,
    doppler_mask_max: int,
    # occlusion
    occlusion_enable: bool,
    occlusion_prob: float,
    occlusion_range_band: tuple[int, int],
    occlusion_doppler_band: tuple[int, int],
    occlusion_attenuation: tuple[float, float],
):
    x = x.copy()

    # gain
    x *= rng.uniform(0.85, 1.15)

    # noise
    x += rng.normal(0.0, 0.08, size=x.shape).astype(np.float32)

    # roll shift
    dy = rng.integers(-3, 4)
    dx = rng.integers(-2, 3)
    x = np.roll(x, shift=int(dy), axis=0)
    x = np.roll(x, shift=int(dx), axis=1)

    # masking aug
    if use_masking_aug:
        if rng.random() < mask_prob:
            H = x.shape[0]
            mlen = int(rng.integers(range_mask_min, max(range_mask_min + 1, min(range_mask_max, H)) + 1))
            start = int(rng.integers(0, max(1, H - mlen + 1)))
            x[start:start + mlen, :, :] = 0.0

        if rng.random() < mask_prob:
            W = x.shape[1]
            mlen = int(rng.integers(doppler_mask_min, max(doppler_mask_min + 1, min(doppler_mask_max, W)) + 1))
            start = int(rng.integers(0, max(1, W - mlen + 1)))
            x[:, start:start + mlen, :] = 0.0

    # ✅ Occlusion (after noise/shift/masking)
    if occlusion_enable:
        x = apply_numpy_occlusion(
            x,
            rng=rng,
            prob=float(occlusion_prob),
            range_band=occlusion_range_band,
            doppler_band=occlusion_doppler_band,
            attenuation=occlusion_attenuation,
        )

    return x.astype(np.float32, copy=False)


# ==========================================================
# PATCH #3 helper: apply occlusion to entire arrays (val/test)
# ==========================================================
def maybe_apply_occlusion_to_array(
    X: np.ndarray,
    seed: int,
    enable: bool,
    prob: float,
    range_band: tuple[int, int],
    doppler_band: tuple[int, int],
    attenuation: tuple[float, float],
) -> np.ndarray:
    if not enable or float(prob) <= 0.0:
        return X
    rng = np.random.default_rng(int(seed))
    Xo = X.copy().astype(np.float32, copy=False)
    for i in range(len(Xo)):
        Xo[i] = apply_numpy_occlusion(
            Xo[i],
            rng=rng,
            prob=float(prob),
            range_band=range_band,
            doppler_band=doppler_band,
            attenuation=attenuation,
        )
    return Xo


# -------------------------
# Keras Sequence (NO tf.data)
# -------------------------
class RadarSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        use_aug: bool,
        use_masking_aug: bool,
        mask_prob: float,
        range_mask_min: int,
        range_mask_max: int,
        doppler_mask_min: int,
        doppler_mask_max: int,
        # occlusion
        occlusion_enable: bool,
        occlusion_prob: float,
        occlusion_range_band: tuple[int, int],
        occlusion_doppler_band: tuple[int, int],
        occlusion_attenuation: tuple[float, float],
        # ✅ PATCH #1: Keras 3 compatibility (removes warning)
        **kwargs,
    ):
        super().__init__(**kwargs)  # ✅ required by Keras 3 PyDataset adapter

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y_onehot, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.use_aug = bool(use_aug)

        self.use_masking_aug = bool(use_masking_aug)
        self.mask_prob = float(mask_prob)
        self.range_mask_min = int(range_mask_min)
        self.range_mask_max = int(range_mask_max)
        self.doppler_mask_min = int(doppler_mask_min)
        self.doppler_mask_max = int(doppler_mask_max)

        self.occlusion_enable = bool(occlusion_enable)
        self.occlusion_prob = float(occlusion_prob)
        self.occlusion_range_band = (int(occlusion_range_band[0]), int(occlusion_range_band[1]))
        self.occlusion_doppler_band = (int(occlusion_doppler_band[0]), int(occlusion_doppler_band[1]))
        self.occlusion_attenuation = (float(occlusion_attenuation[0]), float(occlusion_attenuation[1]))

        self.idx = np.arange(len(self.X))
        self.rng = np.random.default_rng(self.seed)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def __getitem__(self, i):
        sl = self.idx[i * self.batch_size:(i + 1) * self.batch_size]
        xb = self.X[sl]
        yb = self.y[sl]

        if self.use_aug:
            out = np.empty_like(xb)
            for k in range(len(xb)):
                out[k] = apply_numpy_aug(
                    xb[k],
                    rng=self.rng,
                    use_masking_aug=self.use_masking_aug,
                    mask_prob=self.mask_prob,
                    range_mask_min=self.range_mask_min,
                    range_mask_max=self.range_mask_max,
                    doppler_mask_min=self.doppler_mask_min,
                    doppler_mask_max=self.doppler_mask_max,
                    occlusion_enable=self.occlusion_enable,
                    occlusion_prob=self.occlusion_prob,
                    occlusion_range_band=self.occlusion_range_band,
                    occlusion_doppler_band=self.occlusion_doppler_band,
                    occlusion_attenuation=self.occlusion_attenuation,
                )
            xb = out

        return xb, yb


# -------------------------
# Model
# -------------------------
def build_cnn(input_shape, n_classes, lr, label_smoothing):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(24, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(48, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(96, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(192, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(lr)),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# -------------------------
# Rejection Threshold Evaluation
# -------------------------
def evaluate_with_rejection(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    threshold: float,
):
    """
    pmax < T => reject (-1)
    coverage = accepted%
    acc_accepted = accuracy on accepted only
    also returns reject counts per true class and cm_with_reject
    """
    T = float(threshold)
    pmax = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1).astype(int)

    reject_mask = pmax < T
    y_pred_with_reject = y_pred.copy()
    y_pred_with_reject[reject_mask] = -1

    coverage = 1.0 - float(np.mean(reject_mask))

    if np.any(~reject_mask):
        acc_accepted = float(np.mean(y_pred[~reject_mask] == y_true[~reject_mask]))
    else:
        acc_accepted = 0.0

    rej_counts = reject_counts_by_true_class(y_true, reject_mask, class_names)
    cm_wr = confusion_matrix_with_reject(y_true, y_pred_with_reject, n_classes=len(class_names))

    return {
        "threshold": T,
        "coverage": coverage,
        "acc_accepted": acc_accepted,
        "total": int(len(y_true)),
        "accepted": int(np.sum(~reject_mask)),
        "rejected": int(np.sum(reject_mask)),
        "reject_counts_by_true_class": rej_counts,
        "confusion_matrix_with_reject": cm_wr,
    }


def save_rejection_sweep_csv(run_dir: Path, sweep_rows: list[dict]):
    """
    Saves: threshold, coverage, acc_accepted, accepted, rejected
    """
    out_path = run_dir / "rejection_sweep.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("threshold,coverage,acc_accepted,accepted,rejected\n")
        for r in sweep_rows:
            f.write(
                f"{r['threshold']:.2f},{r['coverage']:.6f},{r['acc_accepted']:.6f},{r['accepted']},{r['rejected']}\n"
            )
    return out_path


# -------------------------
# Main
# -------------------------
def main(config_path: str) -> None:
    cfg = load_yaml(config_path)

    seed = safe_int(cfg.get("seed", 42), 42)
    set_seed(seed)
    configure_tf_runtime(cfg)

    out_dir = ensure_dir(cfg.get("paths", {}).get("out_dir", "outputs"))
    run_dir = ensure_dir(out_dir / f"cnn_run_{time.strftime('%Y%m%d_%H%M%S')}")

    test_size = safe_float(cfg.get("data", {}).get("test_size", 0.2), 0.2)
    val_size = safe_float(cfg.get("data", {}).get("val_size", 0.1), 0.1)

    cnn_cfg = cfg.get("cnn", {})
    epochs = safe_int(cnn_cfg.get("epochs", 30), 30)
    batch_size = safe_int(cnn_cfg.get("batch_size", 32), 32)
    lr = safe_float(cnn_cfg.get("lr", 1e-3), 1e-3)
    label_smoothing = safe_float(cnn_cfg.get("label_smoothing", 0.03), 0.03)

    safe_cap = safe_int(cfg.get("safe_batch_cap", 16), 16)
    if safe_cap > 0 and batch_size > safe_cap:
        print(f"🛟 safe_batch_cap={safe_cap} → reducing batch_size {batch_size} → {safe_cap}", flush=True)
        batch_size = safe_cap

    use_aug = safe_bool(cfg.get("use_augmentation", True), True)
    aug_cfg = cfg.get("augmentation", {}) if isinstance(cfg.get("augmentation", {}), dict) else {}
    use_masking_aug = safe_bool(aug_cfg.get("use_masking_aug", True), True)
    mask_prob = safe_float(aug_cfg.get("mask_prob", 0.35), 0.35)
    range_mask_min = safe_int(aug_cfg.get("range_mask_min", 3), 3)
    range_mask_max = safe_int(aug_cfg.get("range_mask_max", 10), 10)
    doppler_mask_min = safe_int(aug_cfg.get("doppler_mask_min", 1), 1)
    doppler_mask_max = safe_int(aug_cfg.get("doppler_mask_max", 4), 4)

    # ✅ Occlusion config
    occ_cfg = cfg.get("occlusion", {}) if isinstance(cfg.get("occlusion", {}), dict) else {}
    occlusion_enable = safe_bool(occ_cfg.get("enable", False), False)
    occlusion_prob = safe_float(occ_cfg.get("prob", 0.30), 0.30)
    occlusion_range_band = occ_cfg.get("range_band", [12, 40])
    occlusion_doppler_band = occ_cfg.get("doppler_band", [2, 10])
    occlusion_attenuation = occ_cfg.get("attenuation", [0.0, 0.4])

    # ✅ PATCH #3: mode support
    occlusion_mode = str(occ_cfg.get("mode", "train")).strip().lower()
    if occlusion_mode not in ("train", "train+val", "train+val+test"):
        occlusion_mode = "train"

    if not (isinstance(occlusion_range_band, (list, tuple)) and len(occlusion_range_band) == 2):
        occlusion_range_band = [12, 40]
    if not (isinstance(occlusion_doppler_band, (list, tuple)) and len(occlusion_doppler_band) == 2):
        occlusion_doppler_band = [2, 10]
    if not (isinstance(occlusion_attenuation, (list, tuple)) and len(occlusion_attenuation) == 2):
        occlusion_attenuation = [0.0, 0.4]

    occlusion_range_band = (safe_int(occlusion_range_band[0], 12), safe_int(occlusion_range_band[1], 40))
    occlusion_doppler_band = (safe_int(occlusion_doppler_band[0], 2), safe_int(occlusion_doppler_band[1], 10))
    occlusion_attenuation = (safe_float(occlusion_attenuation[0], 0.0), safe_float(occlusion_attenuation[1], 0.4))

    print(
        f"🧱 Occlusion: enable={occlusion_enable} mode={occlusion_mode} prob={occlusion_prob} "
        f"range_band={occlusion_range_band} doppler_band={occlusion_doppler_band} att={occlusion_attenuation}",
        flush=True
    )

    # ✅ Rejection threshold config
    rej_cfg = cfg.get("rejection", {}) if isinstance(cfg.get("rejection", {}), dict) else {}
    rejection_enable = safe_bool(rej_cfg.get("enable", True), True)
    rejection_threshold = safe_float(rej_cfg.get("threshold", 0.75), 0.75)
    print(f"🛑 Rejection: enable={rejection_enable} threshold={rejection_threshold}", flush=True)

    class_names = cfg.get("class_names", ["car", "pedestrian", "cyclist", "drone", "truck", "tank"])
    if not isinstance(class_names, list) or len(class_names) < 2:
        class_names = ["car", "pedestrian", "cyclist", "drone", "truck", "tank"]
    for req in ["car", "pedestrian", "cyclist", "drone", "truck", "tank"]:
        if req not in class_names:
            class_names.append(req)
    n_classes = len(class_names)

    carrada_root = cfg.get("carrada_root", "data/raw/carrada")
    carrada_limit = cfg.get("carrada_limit", None)
    carrada_limit = safe_int(carrada_limit, 0) if carrada_limit is not None else None
    carrada_use = str(cfg.get("carrada_use", "rd"))
    carrada_single_object_only = safe_bool(cfg.get("carrada_single_object_only", False), False)

    balance_train_only = safe_bool(cfg.get("balance_train_only", True), True)
    train_min_per_class = safe_int(cfg.get("train_min_per_class", 800), 800)
    train_max_per_class = safe_int(cfg.get("train_max_per_class", 2400), 2400)
    oversample_cap_factor = safe_int(cfg.get("oversample_cap_factor", 200), 200)

    print(f"📡 Loading Carrada from: {carrada_root}", flush=True)
    print(
        f"🧪 Carrada options: use={carrada_use}, single_object_only={carrada_single_object_only}",
        flush=True,
    )

    X_c, y_c = load_carrada_dataset(
        root=carrada_root,
        limit=carrada_limit,
        seed=seed,
        use=carrada_use,
        single_object_only=carrada_single_object_only,
        balanced=False,
        min_per_class=None,
        max_per_class=None,
        debug=True,
    )

    if X_c.size == 0 or len(y_c) == 0:
        raise RuntimeError("❌ Loaded 0 Carrada samples. Check carrada_root path + dataset structure.")

    X_c = normalize_per_sample(X_c)
    if X_c.ndim == 4 and X_c.shape[1] == 1:
        X_c = np.transpose(X_c, (0, 2, 3, 1))  # (N,H,W,1)

    H, W, C = int(X_c.shape[1]), int(X_c.shape[2]), int(X_c.shape[3])

    print(f"✅ Carrada loaded X shape: {X_c.shape}  (H={H}, W={W}, C={C})", flush=True)
    print_class_counts("ALL (Carrada only, unbalanced)", y_c, ["car", "pedestrian", "cyclist"])

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_np(
        X_c, y_c, test_size=test_size, val_size=val_size, seed=seed
    )
    print(f"✅ Split sizes (Carrada): train={len(X_train)} val={len(X_val)} test={len(X_test)}", flush=True)
    print_class_counts("TRAIN (Carrada)", y_train, ["car", "pedestrian", "cyclist"])
    print_class_counts("VAL (Carrada)", y_val, ["car", "pedestrian", "cyclist"])
    print_class_counts("TEST (Carrada)", y_test, ["car", "pedestrian", "cyclist"])

    # ✅ PATCH #3: apply occlusion to val/test if requested
    if occlusion_enable and occlusion_mode in ("train+val", "train+val+test"):
        print("🧱 Applying occlusion to VAL split (mode enabled)...", flush=True)
        X_val = maybe_apply_occlusion_to_array(
            X_val,
            seed=seed + 101,
            enable=True,
            prob=occlusion_prob,
            range_band=occlusion_range_band,
            doppler_band=occlusion_doppler_band,
            attenuation=occlusion_attenuation,
        )

    if occlusion_enable and occlusion_mode == "train+val+test":
        print("🧱 Applying occlusion to TEST split (mode enabled)...", flush=True)
        X_test = maybe_apply_occlusion_to_array(
            X_test,
            seed=seed + 202,
            enable=True,
            prob=occlusion_prob,
            range_band=occlusion_range_band,
            doppler_band=occlusion_doppler_band,
            attenuation=occlusion_attenuation,
        )

    if balance_train_only:
        print("⚖️ Balancing TRAIN only (Carrada) using balance_xy() ...", flush=True)
        X_train, y_train = balance_xy(
            X_train,
            y_train,
            seed=seed,
            min_per_class=train_min_per_class,
            max_per_class=train_max_per_class,
            oversample_cap_factor=oversample_cap_factor,
            debug=True,
        )
        print_class_counts("TRAIN (Carrada after balance)", y_train, ["car", "pedestrian", "cyclist"])

    # ✅ PATCH #2: YAML compat (use_sim_data OR use_sim)
    use_sim = safe_bool(cfg.get("use_sim_data", cfg.get("use_sim", True)), True)
    sim_root = cfg.get("sim_root", "matlab/04_train_with_sim_data/sim_data_npy")
    sim_limit_per_class = cfg.get("sim_limit_per_class", None)
    sim_limit_per_class = safe_int(sim_limit_per_class, 0) if sim_limit_per_class is not None else None
    if sim_limit_per_class is not None and sim_limit_per_class <= 0:
        sim_limit_per_class = None

    # holders for SIM eval
    X_sim_test, y_sim_test = None, None

    if use_sim:
        print(f"🧪 Loading MATLAB synthetic RD from: {sim_root}", flush=True)
        X_sim, y_sim = load_matlab_synthetic_rd(
            sim_root=sim_root,
            class_names=class_names,
            input_shape=(H, W, C),
            limit_per_class=sim_limit_per_class,
            seed=seed,
            strict_shape=True,
            verbose=True,
        )
        print_class_counts("SIM (MATLAB)", y_sim, class_names)

        # ✅ PATCH #4: SIM split so we can evaluate defence targets properly
        X_sim_train, y_sim_train, X_sim_val, y_sim_val, X_sim_test, y_sim_test = train_val_test_split_np(
            X_sim, y_sim, test_size=test_size, val_size=val_size, seed=seed
        )

        print(f"✅ SIM split sizes: train={len(X_sim_train)} val={len(X_sim_val)} test={len(X_sim_test)}", flush=True)
        print_class_counts("SIM TRAIN", y_sim_train, class_names)
        print_class_counts("SIM TEST", y_sim_test, class_names)

        # merge SIM TRAIN into training only
        X_train = np.concatenate([X_train, X_sim_train], axis=0)
        y_train = np.concatenate([y_train, y_sim_train], axis=0)
        print(f"✅ TRAIN after adding MATLAB(SIM TRAIN): X_train={X_train.shape} y_train={y_train.shape}", flush=True)
        print_class_counts("TRAIN (after adding SIM TRAIN)", y_train, class_names)

    # one-hot
    y_train_oh = to_one_hot(y_train, n_classes)
    y_val_oh = to_one_hot(y_val, n_classes) if len(y_val) > 0 else None
    y_test_oh = to_one_hot(y_test, n_classes)

    # model
    model = build_cnn(input_shape=(H, W, C), n_classes=n_classes, lr=lr, label_smoothing=label_smoothing)
    model.summary()

    # ✅ Sequence generator (no tf.data)
    train_seq = RadarSequence(
        X_train, y_train_oh,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        use_aug=use_aug,
        use_masking_aug=use_masking_aug,
        mask_prob=mask_prob,
        range_mask_min=range_mask_min,
        range_mask_max=range_mask_max,
        doppler_mask_min=doppler_mask_min,
        doppler_mask_max=doppler_mask_max,
        occlusion_enable=(occlusion_enable and occlusion_mode.startswith("train")),
        occlusion_prob=occlusion_prob,
        occlusion_range_band=occlusion_range_band,
        occlusion_doppler_band=occlusion_doppler_band,
        occlusion_attenuation=occlusion_attenuation,
    )

    # callbacks
    best_path = run_dir / "best_model.keras"
    cb = []
    if y_val_oh is not None:
        cb = [
            callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
            callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=7, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ]

    print("🧪 Sanity-check: getting 1 batch from train_seq...", flush=True)
    xb, yb = train_seq[0]
    print("✅ Got batch:", xb.shape, yb.shape, xb.dtype, yb.dtype, flush=True)

    print("🚀 Starting training now...", flush=True)

    # NOTE (Keras 3): do NOT pass workers/use_multiprocessing/max_queue_size to fit() here.
    hist = model.fit(
        train_seq,
        epochs=epochs,
        validation_data=(X_val, y_val_oh) if y_val_oh is not None else None,
        callbacks=cb,
        verbose=1,
    )

    # save history
    hist_path = run_dir / "history.csv"
    if hist is not None and hasattr(hist, "history"):
        keys = list(hist.history.keys())
        with open(hist_path, "w", encoding="utf-8") as f:
            f.write("epoch," + ",".join(keys) + "\n")
            for i in range(len(hist.history[keys[0]])):
                row = [str(i + 1)] + [str(float(hist.history[k][i])) for k in keys]
                f.write(",".join(row) + "\n")

        # ✅ NEW: Save history dict as .npy for figure generator
        np.save(run_dir / "history.npy", hist.history, allow_pickle=True)
        print("✅ Saved history.npy for IEEE figures", flush=True)

    # eval (load best)
    if best_path.exists():
        model = tf.keras.models.load_model(best_path, compile=True)

    # -------------------------
    # Carrada test evaluation
    # -------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"🧪 Carrada TEST loss={test_loss:.4f}  acc={test_acc:.4f}", flush=True)

    probs_test = model.predict(X_test, verbose=0)
    np.save(run_dir / "probs_test.npy", probs_test)

    y_pred = np.argmax(probs_test, axis=1)

    # ✅ NEW: Save arrays needed for IEEE figures
    np.save(run_dir / "X_test.npy", X_test)
    np.save(run_dir / "y_test.npy", y_test)
    np.save(run_dir / "y_pred.npy", y_pred)
    print("✅ Saved figure artifacts: X_test.npy, y_test.npy, y_pred.npy, probs_test.npy", flush=True)

    cm = confusion_matrix_np(y_test, y_pred, n_classes=n_classes)
    report_named = classification_report_named(cm, class_names)
    pred_counts = predicted_distribution(y_pred, class_names)

    print("\nCarrada Confusion Matrix (rows=true, cols=pred):", flush=True)
    print(cm, flush=True)
    print(f"\n🔎 Carrada TEST predicted distribution: {pred_counts}", flush=True)

    # ✅ Rejection threshold evaluation + sweep (on Carrada test by default)
    rejection_results = None
    sweep_rows = []
    sweep_csv_path = None

    if rejection_enable:
        r_main = evaluate_with_rejection(
            probs=probs_test,
            y_true=y_test,
            class_names=class_names,
            threshold=rejection_threshold,
        )
        rejection_results = {
            "main_threshold": {
                "threshold": r_main["threshold"],
                "coverage": r_main["coverage"],
                "acc_accepted": r_main["acc_accepted"],
                "accepted": r_main["accepted"],
                "rejected": r_main["rejected"],
                "reject_counts_by_true_class": r_main["reject_counts_by_true_class"],
                "confusion_matrix_with_reject": r_main["confusion_matrix_with_reject"].tolist(),
            }
        }

        print("\n🛑 Rejection (Carrada main):", flush=True)
        print(
            f"  T={r_main['threshold']:.2f} | coverage={r_main['coverage']:.3f} | acc_accepted={r_main['acc_accepted']:.3f}",
            flush=True
        )
        print(f"  accepted={r_main['accepted']} rejected={r_main['rejected']}", flush=True)

        sweep_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
        for T in sweep_thresholds:
            r = evaluate_with_rejection(
                probs=probs_test,
                y_true=y_test,
                class_names=class_names,
                threshold=T,
            )
            sweep_rows.append({
                "threshold": r["threshold"],
                "coverage": r["coverage"],
                "acc_accepted": r["acc_accepted"],
                "accepted": r["accepted"],
                "rejected": r["rejected"],
                "reject_counts_by_true_class": r["reject_counts_by_true_class"],
            })

        sweep_csv_path = save_rejection_sweep_csv(run_dir, sweep_rows)
        rejection_results["sweep"] = {
            "thresholds": sweep_thresholds,
            "rows": sweep_rows,
            "csv": str(sweep_csv_path),
        }

        print("\n📉 Rejection sweep saved:", flush=True)
        print(f"  {sweep_csv_path}", flush=True)

    # -------------------------
    # ✅ PATCH #4: SIM test evaluation (defence targets)
    # -------------------------
    sim_metrics = None
    if use_sim and X_sim_test is not None and y_sim_test is not None and len(X_sim_test) > 0:
        y_sim_test_oh = to_one_hot(y_sim_test, n_classes)
        sim_loss, sim_acc = model.evaluate(X_sim_test, y_sim_test_oh, verbose=0)
        print(f"\n🧪 SIM TEST (defence targets) loss={sim_loss:.4f}  acc={sim_acc:.4f}", flush=True)

        probs_sim = model.predict(X_sim_test, verbose=0)
        np.save(run_dir / "probs_sim_test.npy", probs_sim)

        y_sim_pred = np.argmax(probs_sim, axis=1)
        cm_sim = confusion_matrix_np(y_sim_test, y_sim_pred, n_classes=n_classes)
        sim_pred_counts = predicted_distribution(y_sim_pred, class_names)

        print("\nSIM Confusion Matrix (rows=true, cols=pred):", flush=True)
        print(cm_sim, flush=True)
        print(f"\n🔎 SIM TEST predicted distribution: {sim_pred_counts}", flush=True)

        sim_metrics = {
            "sim_loss": float(sim_loss),
            "sim_acc": float(sim_acc),
            "sim_confusion_matrix": cm_sim.tolist(),
            "sim_predicted_distribution": sim_pred_counts,
        }

    # save model
    model.save(run_dir / "final_model.keras")

    # save metrics
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "carrada_test_loss": float(test_loss),
                "carrada_test_acc": float(test_acc),
                "class_names": class_names,
                "data_shape": [H, W, C],
                "predicted_distribution_carrada_test": pred_counts,
                "confusion_matrix_carrada_test": cm.tolist(),
                "report_named_carrada_test": report_named,
                "occlusion": {
                    "enable": bool(occlusion_enable),
                    "mode": occlusion_mode,
                    "prob": float(occlusion_prob),
                    "range_band": list(occlusion_range_band),
                    "doppler_band": list(occlusion_doppler_band),
                    "attenuation": list(occlusion_attenuation),
                },
                "rejection": {
                    "enable": bool(rejection_enable),
                    "threshold": float(rejection_threshold),
                    "results": rejection_results,
                },
                "sim_eval": sim_metrics,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved outputs to: {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/experiment.yml")
    args = parser.parse_args()
    main(args.config)
