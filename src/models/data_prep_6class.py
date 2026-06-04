from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.loaders import balance_xy, load_carrada_dataset


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


def safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def normalize_per_sample(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    if X.ndim != 4:
        raise RuntimeError(f"Expected 4D array, got shape={X.shape}")
    if X.shape[1] == 1:
        for i in range(X.shape[0]):
            xi = X[i, 0]
            m = float(xi.mean())
            s = float(xi.std())
            if s < eps:
                s = eps
            X[i, 0] = (xi - m) / (s + eps)
        return X
    if X.shape[-1] == 1:
        for i in range(X.shape[0]):
            xi = X[i, :, :, 0]
            m = float(xi.mean())
            s = float(xi.std())
            if s < eps:
                s = eps
            X[i, :, :, 0] = (xi - m) / (s + eps)
        return X
    raise RuntimeError(f"Unexpected channel layout: {X.shape}")


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


def load_matlab_synthetic_rd(sim_root, class_names, input_shape, limit_per_class=None, seed=42):
    sim_root = Path(sim_root)
    rng = np.random.default_rng(seed)
    sim_classes = ["drone", "truck", "tank"]
    H, W, C = input_shape

    X_list, y_list = [], []
    for cls in sim_classes:
        cls_dir = sim_root / cls
        files = sorted(cls_dir.glob("*.npy")) if cls_dir.exists() else []
        if len(files) == 0:
            continue
        if limit_per_class and len(files) > limit_per_class:
            idx = rng.permutation(len(files))[:limit_per_class]
            files = [files[i] for i in idx]

        cls_idx = class_names.index(cls)
        for fp in files:
            rd = np.load(fp)
            if rd.ndim == 2:
                rd = rd[..., np.newaxis]
            rd = rd.astype(np.float32, copy=False)
            if rd.shape != (H, W, C):
                continue
            X_list.append(rd)
            y_list.append(cls_idx)

    if len(X_list) == 0:
        raise RuntimeError(f"No MATLAB synthetic RD samples found under: {sim_root}")

    X_sim = np.stack(X_list, axis=0)
    y_sim = np.array(y_list, dtype=np.int64)
    return X_sim, y_sim


def _shuffle_pair(X: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def load_joint_6class_dataset(cfg: dict) -> dict:
    seed = safe_int(cfg.get("seed", 42), 42)
    class_names = cfg.get("class_names", ["car", "pedestrian", "cyclist", "drone", "truck", "tank"])

    carrada_root = cfg.get("carrada_root", "data/raw/carrada")
    carrada_limit = cfg.get("carrada_limit", None)
    carrada_limit = safe_int(carrada_limit, 0) if carrada_limit is not None else None
    carrada_use = str(cfg.get("carrada_use", "rd"))
    single_object_only = safe_bool(cfg.get("carrada_single_object_only", False), False)
    test_size = float(cfg.get("data", {}).get("test_size", 0.2))
    val_size = float(cfg.get("data", {}).get("val_size", 0.1))

    X_c, y_c = load_carrada_dataset(
        root=carrada_root,
        limit=carrada_limit,
        seed=seed,
        use=carrada_use,
        single_object_only=single_object_only,
        balanced=False,
        min_per_class=None,
        max_per_class=None,
        debug=False,
    )
    X_c = normalize_per_sample(X_c)
    if X_c.ndim == 4 and X_c.shape[1] == 1:
        X_c = np.transpose(X_c, (0, 2, 3, 1))

    X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c = train_val_test_split_np(
        X_c, y_c, test_size=test_size, val_size=val_size, seed=seed
    )

    if safe_bool(cfg.get("balance_train_only", True), True):
        X_train_c, y_train_c = balance_xy(
            X_train_c,
            y_train_c,
            seed=seed,
            min_per_class=safe_int(cfg.get("train_min_per_class", 800), 800),
            max_per_class=safe_int(cfg.get("train_max_per_class", 2400), 2400),
            oversample_cap_factor=safe_int(cfg.get("oversample_cap_factor", 200), 200),
            debug=False,
        )

    H, W, C = int(X_c.shape[1]), int(X_c.shape[2]), int(X_c.shape[3])
    use_sim = safe_bool(cfg.get("use_sim_data", cfg.get("use_sim", True)), True)

    if use_sim:
        sim_root = cfg.get("sim_root", "matlab/04_train_with_sim_data/sim_data_npy")
        sim_limit = cfg.get("sim_limit_per_class", None)
        sim_limit = safe_int(sim_limit, 0) if sim_limit is not None else None
        if sim_limit is not None and sim_limit <= 0:
            sim_limit = None

        X_sim, y_sim = load_matlab_synthetic_rd(
            sim_root=sim_root,
            class_names=class_names,
            input_shape=(H, W, C),
            limit_per_class=sim_limit,
            seed=seed,
        )
        X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = train_val_test_split_np(
            X_sim, y_sim, test_size=test_size, val_size=val_size, seed=seed
        )
    else:
        X_train_s = np.empty((0, H, W, C), dtype=np.float32)
        y_train_s = np.empty((0,), dtype=np.int64)
        X_val_s = np.empty((0, H, W, C), dtype=np.float32)
        y_val_s = np.empty((0,), dtype=np.int64)
        X_test_s = np.empty((0, H, W, C), dtype=np.float32)
        y_test_s = np.empty((0,), dtype=np.int64)

    X_train = np.concatenate([X_train_c, X_train_s], axis=0)
    y_train = np.concatenate([y_train_c, y_train_s], axis=0)
    X_val = np.concatenate([X_val_c, X_val_s], axis=0)
    y_val = np.concatenate([y_val_c, y_val_s], axis=0)
    X_test = np.concatenate([X_test_c, X_test_s], axis=0)
    y_test = np.concatenate([y_test_c, y_test_s], axis=0)

    X_train, y_train = _shuffle_pair(X_train, y_train, seed + 11)
    X_val, y_val = _shuffle_pair(X_val, y_val, seed + 12)
    X_test, y_test = _shuffle_pair(X_test, y_test, seed + 13)

    return {
        "class_names": class_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_carrada_test": X_test_c,
        "y_carrada_test": y_test_c,
        "X_sim_test": X_test_s,
        "y_sim_test": y_test_s,
    }
