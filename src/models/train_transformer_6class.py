from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models

from src.models.data_prep_6class import load_joint_6class_dataset


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_transformer(input_shape, n_classes, patch_h=16, patch_w=8, d_model=64, num_heads=4, depth=3, mlp_dim=128):
    h, w, c = input_shape
    if h % patch_h != 0 or w % patch_w != 0:
        raise ValueError("Input shape must be divisible by patch sizes")

    num_patches = (h // patch_h) * (w // patch_w)
    patch_dim = patch_h * patch_w * c

    inp = layers.Input(shape=input_shape)

    patches = layers.Lambda(
        lambda x: tf.image.extract_patches(
            images=x,
            sizes=[1, patch_h, patch_w, 1],
            strides=[1, patch_h, patch_w, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
    )(inp)
    x = layers.Reshape((num_patches, patch_dim))(patches)
    x = layers.Dense(d_model)(x)

    pos = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=d_model)(pos)
    x = x + pos_embed

    for _ in range(depth):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=0.1)(x1, x1)
        x = layers.Add()([x, attn])
        x2 = layers.LayerNormalization(epsilon=1e-6)(x)
        mlp = layers.Dense(mlp_dim, activation="gelu")(x2)
        mlp = layers.Dropout(0.2)(mlp)
        mlp = layers.Dense(d_model)(mlp)
        mlp = layers.Dropout(0.2)(mlp)
        x = layers.Add()([x, mlp])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main(config_path: str):
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data = load_joint_6class_dataset(cfg)
    X_train, y_train = data["X_train"], data["y_train"].astype(int)
    X_val, y_val = data["X_val"], data["y_val"].astype(int)
    X_test, y_test = data["X_test"], data["y_test"].astype(int)
    X_c_test, y_c_test = data["X_carrada_test"], data["y_carrada_test"].astype(int)
    X_s_test, y_s_test = data["X_sim_test"], data["y_sim_test"].astype(int)

    out_dir = Path(cfg.get("paths", {}).get("out_dir", "outputs"))
    run_dir = out_dir / f"transformer6_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_transformer(input_shape=X_train.shape[1:], n_classes=len(data["class_names"]))

    cls = np.unique(y_train)
    w = compute_class_weight(class_weight="balanced", classes=cls, y=y_train)
    class_weight = {int(c): float(wi) for c, wi in zip(cls, w)}

    best_path = run_dir / "best_model.keras"
    cb = [
        callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )

    # EarlyStopping already restores best weights in-memory.
    # Avoid reloading Lambda-based models with safe deserialization constraints.

    pred_joint = np.argmax(model.predict(X_test, verbose=0), axis=1)
    pred_c = np.argmax(model.predict(X_c_test, verbose=0), axis=1)
    joint_acc = float(accuracy_score(y_test, pred_joint))
    carrada_acc = float(accuracy_score(y_c_test, pred_c))

    sim_acc = None
    if len(X_s_test) > 0:
        pred_s = np.argmax(model.predict(X_s_test, verbose=0), axis=1)
        sim_acc = float(accuracy_score(y_s_test, pred_s))

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": data["class_names"],
                "joint_test_size": int(len(y_test)),
                "carrada_test_size": int(len(y_c_test)),
                "sim_test_size": int(len(y_s_test)),
                "transformer": {
                    "joint_test_acc": joint_acc,
                    "carrada_test_acc": carrada_acc,
                    "sim_test_acc": sim_acc,
                },
            },
            f,
            indent=2,
        )

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2)

    print(
        f"[Transformer] joint={joint_acc:.4f} carrada={carrada_acc:.4f} "
        f"sim={sim_acc if sim_acc is not None else '-'}",
        flush=True,
    )
    print(f"Saved transformer metrics to: {run_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/experiment_sixclass_goal95.yml")
    args = ap.parse_args()
    main(args.config)
