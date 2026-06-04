import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.io import load_yaml, ensure_dir
from src.data.simulate_rcs import simulate_feature_table, save_numpy_splits
from src.features.scale import fit_scaler

def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    out_dir = ensure_dir(cfg["paths"]["out_dir"])
    proc_dir = ensure_dir(cfg["paths"]["processed_dir"])

    X, y = simulate_feature_table(
        n_samples=cfg["data"]["n_samples"],
        n_features=cfg["data"]["n_features"],
        n_classes=cfg["data"]["n_classes"],
        seed=cfg["seed"],
    )
    save_numpy_splits(X, y, proc_dir, seed=cfg["seed"],
                      test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"])

    X_train = np.load(f'{proc_dir}/X_train.npy')
    y_train = np.load(f'{proc_dir}/y_train.npy')
    X_val = np.load(f'{proc_dir}/X_val.npy')
    y_val = np.load(f'{proc_dir}/y_val.npy')
    X_test = np.load(f'{proc_dir}/X_test.npy')
    y_test = np.load(f'{proc_dir}/y_test.npy')

    sc = fit_scaler(X_train)
    X_train, X_val, X_test = sc.transform(X_train), sc.transform(X_val), sc.transform(X_test)
    joblib.dump(sc, out_dir / "scaler.joblib")

    model_name = cfg["classic_ml"]["model"]
    if model_name == "svm":
        params = cfg["classic_ml"]["svm"]
        clf = SVC(**params, probability=True, random_state=cfg["seed"])
    else:
        params = cfg["classic_ml"]["rf"]
        clf = RandomForestClassifier(**params, random_state=cfg["seed"])

    clf.fit(X_train, y_train)

    for split, (X_, y_) in {"val": (X_val, y_val), "test": (X_test, y_test)}.items():
        y_pred = clf.predict(X_)
        acc = accuracy_score(y_, y_pred)
        print(f"{model_name.upper()} {split} accuracy: {acc:.4f}")
        print(classification_report(y_, y_pred))
        cm = confusion_matrix(y_, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix ({split})")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(Path(out_dir) / f"cm_{model_name}_{split}.png", dpi=150)

    joblib.dump(clf, out_dir / f"{model_name}.joblib")
    print(f"Saved model to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
