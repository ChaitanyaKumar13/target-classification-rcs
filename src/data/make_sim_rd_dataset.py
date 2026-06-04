import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # target-classification-rcs/
SIM_SRC = PROJECT_ROOT / "matlab" / "04_train_with_sim_data" / "sim_data_npy"
SIM_DST = PROJECT_ROOT / "data" / "sim_rd"

CLASSES = ["drone", "truck", "tank"]

def main():
    SIM_DST.mkdir(parents=True, exist_ok=True)

    # make folders
    for c in CLASSES:
        (SIM_DST / c).mkdir(parents=True, exist_ok=True)

    copied = 0
    for c in CLASSES:
        src_dir = SIM_SRC / c
        dst_dir = SIM_DST / c

        files = sorted(src_dir.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No .npy files found in {src_dir}")

        for f in files:
            out = dst_dir / f.name
            shutil.copy2(f, out)
            copied += 1

    print("✅ Sim dataset copied")
    print(f"Source: {SIM_SRC}")
    print(f"Dest:   {SIM_DST}")
    print(f"Total files: {copied}")

if __name__ == "__main__":
    main()
