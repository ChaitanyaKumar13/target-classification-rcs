# generate_figures.py
import os
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = r"outputs/cnn_run_20260220_190419"   # ✅ change if your folder name differs
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ["car", "pedestrian", "cyclist", "drone", "truck", "tank"]

def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {path}")

def load_artifacts():
    X_test = np.load(os.path.join(RUN_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(RUN_DIR, "y_test.npy"))
    y_pred = np.load(os.path.join(RUN_DIR, "y_pred.npy"))
    probs_test = np.load(os.path.join(RUN_DIR, "probs_test.npy"))
    return X_test, y_test, y_pred, probs_test

def fig17_ece_score():
    X_test, y_test, y_pred, probs_test = load_artifacts()

    y_true = y_test.astype(int)
    pmax = np.max(probs_test, axis=1)
    pred = np.argmax(probs_test, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    N = len(pmax)

    for i in range(len(bins) - 1):
        mask = (pmax >= bins[i]) & (pmax < bins[i+1])
        if np.sum(mask) == 0:
            continue
        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(pmax[mask])
        ece += (np.sum(mask) / N) * abs(acc_bin - conf_bin)

    plt.figure(figsize=(5, 4))
    plt.bar(["ECE"], [ece])
    plt.ylim(0, max(0.05, ece * 1.5))
    plt.ylabel("Error")
    plt.title(f"Expected Calibration Error (ECE = {ece:.3f})")
    savefig("fig17_ece_score.png")
    plt.show()

if __name__ == "__main__":
    fig17_ece_score()