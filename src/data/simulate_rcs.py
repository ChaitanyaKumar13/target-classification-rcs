import numpy as np
from pathlib import Path
from .split import stratified_split

def simulate_feature_table(n_samples=4000, n_features=32, n_classes=4, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    centers = rng.normal(0, 2.0, size=(n_classes, n_features))
    scales = np.linspace(0.8, 1.8, n_classes)

    for c in range(n_classes):
        n_c = n_samples // n_classes
        cov = np.eye(n_features) * (scales[c] ** 2)
        samples = rng.multivariate_normal(centers[c], cov, size=n_c)
        X.append(samples)
        y.append(np.full(n_c, c))

    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    X += rng.normal(0, 0.25, X.shape) + rng.standard_t(df=5, size=X.shape) * 0.05
    return X, y

def save_numpy_splits(X, y, out_dir: str | Path, seed=42, test_size=0.2, val_size=0.1):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = stratified_split(X, y, test_size=test_size, val_size=val_size, seed=seed)
    for k, (xv, yv) in splits.items():
        np.save(out / f"X_{k}.npy", xv)
        np.save(out / f"y_{k}.npy", yv)
