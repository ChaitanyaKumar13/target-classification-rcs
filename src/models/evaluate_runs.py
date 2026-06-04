import json
from pathlib import Path


RUNS_DIR = Path("outputs")


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def fmt(v, nd=4):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def gather_runs(runs_dir: Path):
    rows = []

    for run_dir in sorted(runs_dir.glob("cnn_run_*")):
        metrics_path = run_dir / "metrics.json"
        best_model = run_dir / "best_model.keras"

        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception as e:
            rows.append({
                "run": run_dir.name,
                "error": str(e),
            })
            continue

        rej = (m.get("rejection") or {}).get("results") or {}
        main_rej = rej.get("main_threshold") or {}

        rows.append(
            {
                "run": run_dir.name,
                "carrada_acc": safe_float(m.get("carrada_test_acc")),
                "carrada_loss": safe_float(m.get("carrada_test_loss")),
                "sim_acc": safe_float((m.get("sim_eval") or {}).get("sim_acc")),
                "coverage": safe_float(main_rej.get("coverage")),
                "acc_accepted": safe_float(main_rej.get("acc_accepted")),
                "rejected": main_rej.get("rejected"),
                "threshold": safe_float(main_rej.get("threshold")),
                "has_best_model": best_model.exists(),
            }
        )

    return rows


def print_table(rows):
    good = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]

    good.sort(
        key=lambda r: (
            r["carrada_acc"] if r["carrada_acc"] is not None else -1.0,
            r["sim_acc"] if r["sim_acc"] is not None else -1.0,
        ),
        reverse=True,
    )

    headers = [
        "rank",
        "run",
        "carrada_acc",
        "sim_acc",
        "cov@T",
        "acc_accpt",
        "rej",
        "T",
        "best_model",
    ]

    lines = []
    for i, r in enumerate(good, start=1):
        lines.append(
            [
                str(i),
                r["run"],
                fmt(r["carrada_acc"]),
                fmt(r["sim_acc"]),
                fmt(r["coverage"]),
                fmt(r["acc_accepted"]),
                str(r["rejected"]) if r["rejected"] is not None else "-",
                fmt(r["threshold"], nd=2),
                "yes" if r["has_best_model"] else "no",
            ]
        )

    widths = [len(h) for h in headers]
    for row in lines:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def render_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(render_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in lines:
        print(render_row(row))

    if good:
        best = good[0]
        print("\nBest run by Carrada accuracy:")
        print(f"  {best['run']} | carrada_acc={fmt(best['carrada_acc'])} | sim_acc={fmt(best['sim_acc'])}")

    if bad:
        print("\nRuns with metrics read errors:")
        for r in bad:
            print(f"  {r['run']}: {r['error']}")


def main():
    rows = gather_runs(RUNS_DIR)
    if not rows:
        print(f"No runs found under: {RUNS_DIR.resolve()}")
        return
    print_table(rows)


if __name__ == "__main__":
    main()
