import argparse
import json
from pathlib import Path


DEFAULT_STAGE_ORDER = [
    "detector_inference_sec",
    "candidate_search_sec",
    "prediction_matching_sec",
    "loss_compute_sec",
    "backpropagation_sec",
    "feature_compute_sec",
]

STAGE_LABELS = {
    "detector_inference_sec": "Detector inference",
    "candidate_search_sec": "Candidate search",
    "prediction_matching_sec": "Prediction matching",
    "loss_compute_sec": "Loss compute",
    "backpropagation_sec": "Backpropagation",
    "feature_compute_sec": "Feature compute",
}

STAGE_COLORS = {
    "detector_inference_sec": "#4C78A8",
    "candidate_search_sec": "#F58518",
    "prediction_matching_sec": "#54A24B",
    "loss_compute_sec": "#E45756",
    "backpropagation_sec": "#B279A2",
    "feature_compute_sec": "#72B7B2",
}


def find_timing_jsons(paths):
    timing_paths = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name.endswith("_timing.json"):
            timing_paths.append(path)
        elif path.is_dir():
            timing_paths.extend(path.rglob("*_timing.json"))
    return sorted(set(p.resolve() for p in timing_paths))


def load_timing_record(path, metric):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = data.get(metric)
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not contain a '{metric}' object.")

    run_dir = path.parent.parent
    uncertainty = str(data.get("uncertainty") or path.name.replace("_timing.json", ""))
    label = f"{uncertainty}\n{run_dir.name}"
    return {
        "path": path,
        "run_dir": run_dir,
        "uncertainty": uncertainty,
        "label": label,
        "values": {str(k): float(v) for k, v in values.items()},
        "total_predictions": int(data.get("total_predictions", 0)),
    }


def ordered_stages(records):
    seen = []
    for stage in DEFAULT_STAGE_ORDER:
        if any(stage in record["values"] for record in records):
            seen.append(stage)
    for record in records:
        for stage in record["values"]:
            if stage not in seen:
                seen.append(stage)
    return seen


def plot_stacked_timing(records, output_path, title, figsize):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to draw timing plots. Install it in the environment used for visualization."
        ) from exc

    if not records:
        raise ValueError("No timing records to plot.")

    stages = ordered_stages(records)
    labels = [record["label"] for record in records]
    x = list(range(len(records)))

    fig_width = max(figsize[0], 1.1 * len(records) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    bottoms = [0.0 for _ in records]

    for stage in stages:
        heights = [record["values"].get(stage, 0.0) for record in records]
        if not any(value > 0.0 for value in heights):
            continue
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=STAGE_LABELS.get(stage, stage),
            color=STAGE_COLORS.get(stage),
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    for idx, total in enumerate(bottoms):
        ax.text(idx, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel("Mean stage time per prediction (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stacked uncertainty timing bars from object_detectors timing JSON files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["object_detectors/runs"],
        help="Run directories, timing directories, or *_timing.json files. Defaults to object_detectors/runs.",
    )
    parser.add_argument(
        "--output",
        default="visualizers/uncertainty_timing_per_prediction.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--metric",
        default="mean_stage_ms_per_prediction",
        choices=["mean_stage_ms_per_prediction", "mean_stage_ms_per_image"],
        help="Timing summary metric to plot.",
    )
    parser.add_argument(
        "--title",
        default="Uncertainty Timing Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.0, 5.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Minimum figure size in inches.",
    )
    parser.add_argument(
        "--uncertainties",
        nargs="*",
        default=None,
        help="Optional uncertainty names to include, e.g. meta_detect layer_grad.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timing_paths = find_timing_jsons(args.paths)
    if not timing_paths:
        searched = ", ".join(str(Path(p)) for p in args.paths)
        raise FileNotFoundError(f"No *_timing.json files found under: {searched}")

    records = [load_timing_record(path, args.metric) for path in timing_paths]
    if args.uncertainties:
        allowed = {name.strip() for name in args.uncertainties}
        records = [record for record in records if record["uncertainty"] in allowed]
    records = sorted(records, key=lambda r: (r["uncertainty"], str(r["run_dir"])))

    if not records:
        raise ValueError("No timing records remain after filtering.")

    output_path = plot_stacked_timing(records, args.output, args.title, tuple(args.figsize))
    print(f"Saved timing plot: {output_path}")


if __name__ == "__main__":
    main()
