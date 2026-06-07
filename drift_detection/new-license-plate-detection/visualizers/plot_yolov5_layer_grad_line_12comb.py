import csv
import html
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

GRID_RESULTS = r"meta_models/meta_classifier/runs/yolov5/train/coco/05-27-2026_13;17_layer_grad_grid/grid_results.csv"

BBOX_LOSSES = ["box_l1", "box_l2"]
CLS_OPTIONS = [
    ("bcewithlogits", "pred_to_target"),
    ("kl", "pred_to_target"),
    ("kl", "target_to_pred"),
]
OBJ_OPTIONS = [
    ("bcewithlogits", "pred_to_target"),
    ("abs_diff", "pred_to_target"),
]

WIDTH = 1100
HEIGHT = 470
PLOT_LEFT = 84
PLOT_TOP = 36
PLOT_RIGHT = 1064
PLOT_BOTTOM = 394
Y_MIN = 0.89
Y_MAX = 0.95
Y_TICKS = [0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_grid_results(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing grid results CSV: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row):
    return (
        row["target"],
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row["obj_loss"],
        row["obj_direction"],
    )


def build_combinations():
    combinations = []
    for bbox_loss in BBOX_LOSSES:
        for cls_loss, cls_direction in CLS_OPTIONS:
            for obj_loss, obj_direction in OBJ_OPTIONS:
                combinations.append(
                    {
                        "bbox_loss": bbox_loss,
                        "bbox_direction": "pred_to_target",
                        "cls_loss": cls_loss,
                        "cls_direction": cls_direction,
                        "obj_loss": obj_loss,
                        "obj_direction": obj_direction,
                    }
                )
    return combinations


def collect_points(rows):
    rows_by_key = {row_key(row): row for row in rows}
    cand_values = []
    null_values = []
    records = []
    for idx, combo in enumerate(build_combinations(), start=1):
        base = (
            combo["bbox_loss"],
            combo["bbox_direction"],
            combo["cls_loss"],
            combo["cls_direction"],
            combo["obj_loss"],
            combo["obj_direction"],
        )
        cand_key = ("cand_target",) + base
        null_key = ("null_target",) + base
        if cand_key not in rows_by_key:
            raise KeyError(f"Missing cand_target row: {cand_key}")
        if null_key not in rows_by_key:
            raise KeyError(f"Missing null_target row: {null_key}")
        cand = float(rows_by_key[cand_key]["auroc"])
        null = float(rows_by_key[null_key]["auroc"])
        cand_values.append(cand)
        null_values.append(null)
        records.append({"idx": idx, "cand_auroc": cand, "null_auroc": null, **combo})
    return cand_values, null_values, records


def x_map(index, count):
    if count <= 1:
        return (PLOT_LEFT + PLOT_RIGHT) / 2
    return PLOT_LEFT + ((index - 1) / (count - 1)) * (PLOT_RIGHT - PLOT_LEFT)


def y_map(value):
    return PLOT_BOTTOM - ((float(value) - Y_MIN) / (Y_MAX - Y_MIN)) * (PLOT_BOTTOM - PLOT_TOP)


def svg_text(x, y, text, size=12, fill="#374151", anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" '
        f'font-size="{size}" fill="{fill}"{extra}>{html.escape(str(text))}</text>'
    )


def path_d(xs, ys):
    parts = [f"M {xs[0]:.1f} {ys[0]:.1f}"]
    parts.extend(f"L {x:.1f} {y:.1f}" for x, y in zip(xs[1:], ys[1:]))
    return " ".join(parts)


def draw_svg(cand_values, null_values, output_path):
    count = len(cand_values)
    xs = [x_map(i, count) for i in range(1, count + 1)]
    cand_ys = [y_map(value) for value in cand_values]
    null_ys = [y_map(value) for value in null_values]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]

    for tick in Y_TICKS:
        y = y_map(tick)
        lines.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(svg_text(72, y + 4, f"{tick:.2f}", fill="#6b7280", anchor="end"))

    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(svg_text(22, 215.0, "AUROC", size=14, fill="#111827", anchor="middle", extra=' transform="rotate(-90 22 215.0)"'))

    for idx, x in enumerate(xs, start=1):
        lines.append(f'<line x1="{x:.1f}" y1="{PLOT_BOTTOM}" x2="{x:.1f}" y2="{PLOT_BOTTOM + 5}" stroke="#111827" stroke-width="1"/>')
        lines.append(svg_text(f"{x:.1f}", 417, idx, fill="#374151", anchor="middle"))
    lines.append(svg_text(574.0, 452, "12 loss combinations", size=14, fill="#111827", anchor="middle"))

    for i in range(count - 1):
        points = (
            f"{xs[i]:.1f},{cand_ys[i]:.1f} "
            f"{xs[i + 1]:.1f},{cand_ys[i + 1]:.1f} "
            f"{xs[i + 1]:.1f},{null_ys[i + 1]:.1f} "
            f"{xs[i]:.1f},{null_ys[i]:.1f}"
        )
        lines.append(f'<polygon points="{points}" fill="#fee2e2" opacity="0.95"/>')

    lines.append(f'<path d="{path_d(xs, cand_ys)}" fill="none" stroke="#334155" stroke-width="2.5"/>')
    lines.append(f'<path d="{path_d(xs, null_ys)}" fill="none" stroke="#dc2626" stroke-width="3.6"/>')

    for x, y in zip(xs, cand_ys):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="#334155" stroke="white" stroke-width="1"/>')
    for x, y in zip(xs, null_ys):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="#dc2626" stroke="white" stroke-width="1"/>')

    lines.append('<line x1="850" y1="34" x2="880" y2="34" stroke="#334155" stroke-width="3"/><circle cx="865" cy="34" r="4" fill="#334155"/>')
    lines.append(svg_text(892, 38, "GradScore", size=14, fill="#111827"))
    lines.append('<line x1="980" y1="34" x2="1010" y2="34" stroke="#dc2626" stroke-width="3.5"/><circle cx="995" cy="34" r="4" fill="#dc2626"/>')
    lines.append(svg_text(1022, 38, "Ours", size=14, fill="#111827"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_records(records, output_path):
    fieldnames = [
        "idx",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "cand_auroc",
        "null_auroc",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def make_output_dir():
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    path = OUTPUT_ROOT / f"{timestamp}_yolov5_layer_grad_line_style_final_12comb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    rows = read_grid_results(GRID_RESULTS)
    cand_values, null_values, records = collect_points(rows)
    output_dir = make_output_dir()
    draw_svg(cand_values, null_values, output_dir / "style_2_clean_12comb_no_signed_obj.svg")
    write_records(records, output_dir / "style_2_clean_12comb_no_signed_obj_points.csv")
    print(f"Saved layer-grad line plot outputs: {output_dir}")


if __name__ == "__main__":
    main()
