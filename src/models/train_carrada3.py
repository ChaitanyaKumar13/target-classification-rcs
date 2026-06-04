from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import callbacks

from src.data.loaders import balance_xy, load_carrada_dataset
from src.models.train_cnn import (
    RadarSequence,
    build_cnn,
    classification_report_named,
    confusion_matrix_np,
    evaluate_with_rejection,
    normalize_per_sample,
    save_rejection_sweep_csv,
    to_one_hot,
    train_val_test_split_np,
)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tf_runtime(cfg: dict):
    tf_cfg = cfg.get("tensorflow", {}) if isinstance(cfg.get("tensorflow", {}), dict) else {}
    intra = int(tf_cfg.get("intra_op_threads", 2))
    inter = int(tf_cfg.get("inter_op_threads", 2))
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    print(f"TF threads configured: intra={intra}, inter={inter}", flush=True)


def main(config_path: str):
    cfg = load_yaml(config_path)
    seed = safe_int(cfg.get("seed", 42), 42)
    set_seed(seed)
    configure_tf_runtime(cfg)

    class_names = ["car", "pedestrian", "cyclist"]
    n_classes = len(class_names)

    out_dir = ensure_dir(cfg.get("paths", {}).get("out_dir", "outputs"))
    run_dir = ensure_dir(out_dir / f"carrada3_run_{time.strftime('%Y%m%d_%H%M%S')}")

    test_size = safe_float(cfg.get("data", {}).get("test_size", 0.2), 0.2)
    val_size = safe_float(cfg.get("data", {}).get("val_size", 0.1), 0.1)

    cnn_cfg = cfg.get("cnn", {})
    epochs = safe_int(cnn_cfg.get("epochs", 50), 50)
    batch_size = safe_int(cnn_cfg.get("batch_size", 16), 16)
    lr = safe_float(cnn_cfg.get("lr", 6e-4), 6e-4)
    label_smoothing = safe_float(cnn_cfg.get("label_smoothing", 0.0), 0.0)

    safe_cap = safe_int(cfg.get("safe_batch_cap", 16), 16)
    if safe_cap > 0 and batch_size > safe_cap:
        batch_size = safe_cap

    carrada_root = cfg.get("carrada_root", "data/raw/carrada")
    carrada_limit = cfg.get("carrada_limit", None)
    carrada_limit = safe_int(carrada_limit, 0) if carrada_limit is not None else None
    carrada_use = str(cfg.get("carrada_use", "rd"))
    carrada_single_object_only = safe_bool(cfg.get("carrada_single_object_only", False), False)

    use_aug = safe_bool(cfg.get("use_augmentation", True), True)
    aug_cfg = cfg.get("augmentation", {}) if isinstance(cfg.get("augmentation", {}), dict) else {}
    use_masking_aug = safe_bool(aug_cfg.get("use_masking_aug", True), True)
    mask_prob = safe_float(aug_cfg.get("mask_prob", 0.15), 0.15)
    range_mask_min = safe_int(aug_cfg.get("range_mask_min", 2), 2)
    range_mask_max = safe_int(aug_cfg.get("range_mask_max", 8), 8)
    doppler_mask_min = safe_int(aug_cfg.get("doppler_mask_min", 1), 1)
    doppler_mask_max = safe_int(aug_cfg.get("doppler_mask_max", 3), 3)

    occ_cfg = cfg.get("occlusion", {}) if isinstance(cfg.get("occlusion", {}), dict) else {}
    occlusion_enable = safe_bool(occ_cfg.get("enable", True), True)
    occlusion_prob = safe_float(occ_cfg.get("prob", 0.1), 0.1)
    occlusion_range_band = occ_cfg.get("range_band", [10, 30])
    occlusion_doppler_band = occ_cfg.get("doppler_band", [2, 8])
    occlusion_attenuation = occ_cfg.get("attenuation", [0.1, 0.5])
    occlusion_mode = str(occ_cfg.get("mode", "train")).strip().lower()

    balance_train_only = safe_bool(cfg.get("balance_train_only", True), True)
    train_min_per_class = safe_int(cfg.get("train_min_per_class", 400), 400)
    train_max_per_class = safe_int(cfg.get("train_max_per_class", 1400), 1400)
    oversample_cap_factor = safe_int(cfg.get("oversample_cap_factor", 100), 100)

    rej_cfg = cfg.get("rejection", {}) if isinstance(cfg.get("rejection", {}), dict) else {}
    rejection_enable = safe_bool(rej_cfg.get("enable", True), True)
    rejection_threshold = safe_float(rej_cfg.get("threshold", 0.75), 0.75)

    print(
        f"Loading Carrada: root={carrada_root}, use={carrada_use}, single_object_only={carrada_single_object_only}",
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
    if X_c.size == 0:
        raise RuntimeError("No Carrada samples loaded.")

    X_c = normalize_per_sample(X_c)
    if X_c.ndim == 4 and X_c.shape[1] == 1:
        X_c = np.transpose(X_c, (0, 2, 3, 1))

    H, W, C = int(X_c.shape[1]), int(X_c.shape[2]), int(X_c.shape[3])
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_np(
        X_c, y_c, test_size=test_size, val_size=val_size, seed=seed
    )
    if balance_train_only:
        X_train, y_train = balance_xy(
            X_train,
            y_train,
            seed=seed,
            min_per_class=train_min_per_class,
            max_per_class=train_max_per_class,
            oversample_cap_factor=oversample_cap_factor,
            debug=True,
        )

    y_train_oh = to_one_hot(y_train, n_classes)
    y_val_oh = to_one_hot(y_val, n_classes)
    y_test_oh = to_one_hot(y_test, n_classes)

    model = build_cnn(input_shape=(H, W, C), n_classes=n_classes, lr=lr, label_smoothing=label_smoothing)

    train_seq = RadarSequence(
        X_train,
        y_train_oh,
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
        occlusion_range_band=(int(occlusion_range_band[0]), int(occlusion_range_band[1])),
        occlusion_doppler_band=(int(occlusion_doppler_band[0]), int(occlusion_doppler_band[1])),
        occlusion_attenuation=(float(occlusion_attenuation[0]), float(occlusion_attenuation[1])),
    )

    best_path = run_dir / "best_model.keras"
    cb = [
        callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    hist = model.fit(
        train_seq,
        epochs=epochs,
        validation_data=(X_val, y_val_oh),
        callbacks=cb,
        verbose=1,
    )

    if best_path.exists():
        model = tf.keras.models.load_model(best_path, compile=True)

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    probs_test = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs_test, axis=1)

    cm = confusion_matrix_np(y_test, y_pred, n_classes=n_classes)
    report_named = classification_report_named(cm, class_names)

    np.save(run_dir / "X_test.npy", X_test)
    np.save(run_dir / "y_test.npy", y_test)
    np.save(run_dir / "y_pred.npy", y_pred)
    np.save(run_dir / "probs_test.npy", probs_test)
    np.save(run_dir / "history.npy", hist.history, allow_pickle=True)

    with open(run_dir / "history.csv", "w", encoding="utf-8") as f:
        keys = list(hist.history.keys())
        f.write("epoch," + ",".join(keys) + "\n")
        for i in range(len(hist.history[keys[0]])):
            row = [str(i + 1)] + [str(float(hist.history[k][i])) for k in keys]
            f.write(",".join(row) + "\n")

    rejection_results = None
    sweep_rows = []
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
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            r = evaluate_with_rejection(
                probs=probs_test,
                y_true=y_test,
                class_names=class_names,
                threshold=t,
            )
            sweep_rows.append(
                {
                    "threshold": r["threshold"],
                    "coverage": r["coverage"],
                    "acc_accepted": r["acc_accepted"],
                    "accepted": r["accepted"],
                    "rejected": r["rejected"],
                    "reject_counts_by_true_class": r["reject_counts_by_true_class"],
                }
            )
        sweep_path = save_rejection_sweep_csv(run_dir, sweep_rows)
        rejection_results["sweep"] = {"rows": sweep_rows, "csv": str(sweep_path)}

    model.save(run_dir / "final_model.keras")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "carrada3_test_loss": float(test_loss),
                "carrada3_test_acc": float(test_acc),
                "class_names": class_names,
                "data_shape": [H, W, C],
                "confusion_matrix_carrada3_test": cm.tolist(),
                "report_named_carrada3_test": report_named,
                "rejection": {
                    "enable": bool(rejection_enable),
                    "threshold": float(rejection_threshold),
                    "results": rejection_results,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved outputs to: {run_dir}", flush=True)
    print(f"Carrada3 TEST acc: {test_acc:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/experiment_carrada3.yml")
    args = parser.parse_args()
    main(args.config)
