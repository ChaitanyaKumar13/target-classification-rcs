# src/data/loaders.py
import json
from pathlib import Path
from collections import Counter

import numpy as np


# Map Carrada class names to our training ids
# Carrada only has: car, pedestrian, cyclist
NAME_TO_TRAIN_ID = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2,
}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_class_id_to_name(seq_dir: Path) -> dict[int, str]:
    """
    Try to discover class-id -> name mapping from per-sequence annotation files.
    If not found, fallback to Carrada convention:
      1=car, 2=pedestrian, 3=cyclist
    """
    ann_dir = seq_dir / "annotations"
    candidates = [
        ann_dir / "instance_oriented.json",
        ann_dir / "frame_oriented.json",
        ann_dir / "annotations_instance_oriented.json",
        ann_dir / "annotations_frame_oriented.json",
    ]

    for p in candidates:
        if not p.exists():
            continue

        d = _load_json(p)

        # common mapping keys
        for key in ("classes", "class", "labels", "label_map", "categories"):
            if key in d and isinstance(d[key], dict):
                m: dict[int, str] = {}
                for k, v in d[key].items():
                    # case: "1": "car"
                    if str(k).isdigit() and isinstance(v, str):
                        m[int(k)] = v.lower().strip()
                    # case: "car": 1
                    if isinstance(k, str) and isinstance(v, int):
                        m[int(v)] = k.lower().strip()
                if m:
                    return m

        # sometimes under metadata
        if "metadata" in d and isinstance(d["metadata"], dict):
            md = d["metadata"]
            for key in ("classes", "labels", "categories"):
                if key in md and isinstance(md[key], dict):
                    m: dict[int, str] = {}
                    for k, v in md[key].items():
                        if str(k).isdigit() and isinstance(v, str):
                            m[int(k)] = v.lower().strip()
                        if isinstance(k, str) and isinstance(v, int):
                            m[int(v)] = k.lower().strip()
                    if m:
                        return m

    return {1: "car", 2: "pedestrian", 3: "cyclist"}


def _choose_one_label(class_ids: list[int]) -> int | None:
    """If multiple objects in a frame, choose the most frequent class-id."""
    if not class_ids:
        return None
    return Counter(class_ids).most_common(1)[0][0]


# -------------------------------------------------------------------
# RAW loader = loads across ALL sequences
# -------------------------------------------------------------------
def _load_raw_carrada(
    root: str,
    limit: int | None = None,
    seed: int = 42,
    use: str = "rd",
    single_object_only: bool = False,
    debug: bool = True,
):
    """
    Loads Carrada RD/RA numpy frames with labels from per-sequence labels.json.

    Behavior:
      - Collect labeled frames across ALL sequences
      - Shuffle globally
      - Apply `limit` at the end

    Returns:
      X: (N, 1, H, W)
      y: (N,)

    single_object_only:
      - False: keep current behavior (choose most frequent class in frame)
      - True: only keep frames that have exactly one labeled object/class-id
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Carrada root not found: {root.resolve()}")

    radar_folder = "range_doppler_numpy" if use.lower() == "rd" else "range_angle_numpy"

    seq_folders = sorted([p for p in root.iterdir() if p.is_dir() and p.name[0].isdigit()])
    if not seq_folders:
        print("No sequence folders found under:", root.resolve())
        return np.array([]), np.array([])

    if debug:
        print(f"[DEBUG] Found {len(seq_folders)} sequence folders under: {root.resolve()}")
        print("[DEBUG] First 5 sequences:", [p.name for p in seq_folders[:5]])

    all_items: list[tuple[Path, int]] = []
    raw_counts = {0: 0, 1: 0, 2: 0}

    used_sequences = 0
    skipped_sequences = 0
    per_seq_counts = {}  # seq -> {0:..,1:..,2:..}

    debug_printed = False

    for seq_dir in seq_folders:
        rd_dir = seq_dir / radar_folder
        label_path = seq_dir / "labels.json"
        if not rd_dir.exists() or not label_path.exists():
            skipped_sequences += 1
            continue

        labels_json = _load_json(label_path)

        # unwrap top-level {"seqname": {...frames...}}
        if isinstance(labels_json, dict) and seq_dir.name in labels_json:
            frame_dict = labels_json[seq_dir.name]
        else:
            frame_dict = labels_json

        if not isinstance(frame_dict, dict):
            skipped_sequences += 1
            continue

        id_to_name = _find_class_id_to_name(seq_dir)
        npy_map = {p.stem: p for p in rd_dir.glob("*.npy")}
        if len(npy_map) == 0:
            skipped_sequences += 1
            continue

        used_sequences += 1
        per_seq_counts[seq_dir.name] = {0: 0, 1: 0, 2: 0}

        # Show ONE sample sequence debug (not “only one sequence used”)
        if debug and not debug_printed:
            some_frames = list(frame_dict.keys())[:5]
            print(f"[DEBUG] Sample seq: {seq_dir.name}")
            print(f"[DEBUG] radar frames in {radar_folder}: {len(npy_map)}")
            print(f"[DEBUG] labels.json frame keys sample: {some_frames}")
            print(f"[DEBUG] class id->name mapping (sample): {dict(list(id_to_name.items())[:5])}")
            debug_printed = True

        for frame_id, obj_to_class in frame_dict.items():
            if frame_id not in npy_map:
                continue

            if not isinstance(obj_to_class, dict) or len(obj_to_class) == 0:
                continue

            class_ids = [cid for cid in obj_to_class.values() if isinstance(cid, int)]
            if single_object_only and len(class_ids) != 1:
                continue
            chosen_class_id = _choose_one_label(class_ids)
            if chosen_class_id is None:
                continue

            class_name = id_to_name.get(chosen_class_id)
            if not isinstance(class_name, str):
                continue

            class_name = class_name.lower().strip()
            if class_name not in NAME_TO_TRAIN_ID:
                continue

            train_id = NAME_TO_TRAIN_ID[class_name]
            all_items.append((npy_map[frame_id], train_id))
            raw_counts[train_id] += 1
            per_seq_counts[seq_dir.name][train_id] += 1

    if debug:
        print(f"[DEBUG] Sequences used: {used_sequences}, skipped: {skipped_sequences}")

    if len(all_items) == 0:
        print("Loaded 0 labeled frames from all sequences.")
        return np.array([]), np.array([])

    if debug:
        print("[DEBUG] Raw labeled counts by class:", raw_counts)

        # Show top 5 sequences by labeled frames
        seq_totals = sorted(
            [(s, sum(cnt.values())) for s, cnt in per_seq_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        print("[DEBUG] Top 5 sequences by labeled frames:")
        for s, tot in seq_totals[:5]:
            print(f"  - {s}: total={tot}, per_class={per_seq_counts[s]}")

    # Global shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(all_items)

    if limit is not None:
        all_items = all_items[: int(limit)]

    X, y = [], []
    for npy_path, lab in all_items:
        arr = np.load(npy_path).astype(np.float32)

        # ensure (1, H, W)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            arr = arr[0:1, :, :]

        X.append(arr)
        y.append(lab)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    if debug:
        u, c = np.unique(y, return_counts=True)
        print("[DEBUG] Final loaded counts (after limit):", dict(zip(u.tolist(), c.tolist())))

    print(f"Loaded {len(X)} radar frames with shape {X.shape[1:]} and {len(np.unique(y))} classes.")
    return X, y


# -------------------------------------------------------------------
# ✅ NEW: Generic balancing utility (for TRAIN ONLY usage)
# -------------------------------------------------------------------
def balance_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    min_per_class: int | None = None,
    max_per_class: int | None = None,
    oversample_cap_factor: int = 200,
    debug: bool = True,
):
    """
    Balance a dataset (X,y) by class, typically used ONLY on TRAIN split.

    - If a class has >= target, it is downsampled without replacement.
    - If a class has < target, it is oversampled with replacement,
      but oversampling is capped to avoid extreme duplication.

    oversample_cap_factor:
      A class with n samples can be oversampled up to n * oversample_cap_factor.
      This prevents a tiny class from being duplicated millions of times.

    Returns:
      X_bal, y_bal
    """
    rng = np.random.default_rng(seed)
    y = y.astype(int)

    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    counts = {int(c): int(len(idx_by_class[c])) for c in classes}
    if debug:
        print("[DEBUG] balance_xy() available counts:", counts)

    if len(counts) == 0:
        return X, y

    # choose base target
    base_target = max(counts.values())
    if max_per_class is not None:
        base_target = min(base_target, int(max_per_class))
    if min_per_class is not None:
        base_target = max(base_target, int(min_per_class))

    chosen_all = []
    for c in classes:
        ix = idx_by_class[c]
        n_avail = len(ix)
        if n_avail == 0:
            continue

        target = base_target
        if max_per_class is not None:
            target = min(target, int(max_per_class))
        if min_per_class is not None:
            target = max(target, int(min_per_class))

        if n_avail >= target:
            chosen = rng.choice(ix, size=target, replace=False)
        else:
            # cap oversampling
            cap = int(n_avail * int(oversample_cap_factor))
            target_eff = min(int(target), max(cap, n_avail))
            chosen = rng.choice(ix, size=target_eff, replace=True)

        chosen_all.append(chosen)

    if not chosen_all:
        return X, y

    chosen_all = np.concatenate(chosen_all, axis=0)
    rng.shuffle(chosen_all)

    X_bal = X[chosen_all]
    y_bal = y[chosen_all]

    if debug:
        u, c = np.unique(y_bal, return_counts=True)
        print("[DEBUG] balance_xy() final balanced counts:", dict(zip(u.tolist(), c.tolist())))

    return X_bal, y_bal


# -------------------------------------------------------------------
# Public loader with optional balancing wrapper (for Carrada only)
# -------------------------------------------------------------------
def load_carrada_dataset(
    root: str = "data/raw/carrada",
    limit: int | None = None,
    seed: int = 42,
    use: str = "rd",                 # "rd" or "ra"
    single_object_only: bool = False,
    balanced: bool = False,
    min_per_class: int | None = None,
    max_per_class: int | None = None,
    debug: bool = True,
):
    """
    Returns:
      X: (N, 1, H, W)
      y: (N,) int labels (0..K-1)

    Uses raw loader across ALL sequences, then (optionally) balances by class.
    NOTE: For best practice, prefer calling load_carrada_dataset(balanced=False),
    then split, then call balance_xy() ONLY on the training split.
    """
    X_raw, y_raw = _load_raw_carrada(
        root=root,
        limit=limit,
        seed=seed,
        use=use,
        single_object_only=single_object_only,
        debug=debug,
    )

    if X_raw is None or len(X_raw) == 0:
        # safe default
        return np.empty((0, 1, 256, 64), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if not balanced:
        return X_raw, y_raw.astype(int)

    # old behavior: balance the whole loaded dataset (not recommended for final eval)
    X_bal, y_bal = balance_xy(
        X_raw,
        y_raw,
        seed=seed,
        min_per_class=min_per_class,
        max_per_class=max_per_class,
        oversample_cap_factor=200,
        debug=debug,
    )
    return X_bal, y_bal.astype(np.int64)
