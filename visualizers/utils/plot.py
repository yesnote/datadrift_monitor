from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


GROUP_PALETTE = {
    "noise": "#888888",
    "fn": "#E45756",
    "non_fn": "#4C78A8",
}


def save_pca_html(
    out_path: Path,
    *,
    points: np.ndarray | None,
    groups: list[str],
    title: str,
    reason: str = "",
    dimension: int = 2,
) -> None:
    import plotly.graph_objects as go

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.update_layout(title=title, legend_title_text="group", template="plotly_white")

    if points is None or points.size == 0:
        msg = "insufficient data" if not reason else f"insufficient data: {reason}"
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    else:
        g_arr = np.asarray(groups, dtype=object)
        unique = [g for g in ["noise", "fn", "non_fn"] if g in set(g_arr.tolist())] or sorted(set(g_arr.tolist()))
        for g in unique:
            mask = g_arr == g
            p = points[mask]
            if p.size == 0:
                continue
            color = GROUP_PALETTE.get(g, "#333333")
            if int(dimension) == 3:
                fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", marker=dict(size=4, color=color, opacity=0.85), name=str(g)))
            else:
                fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="markers", marker=dict(size=7, color=color, opacity=0.85, line=dict(width=0.5, color="black")), name=str(g)))

    if int(dimension) == 3:
        fig.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    else:
        fig.update_xaxes(title_text="PC1")
        fig.update_yaxes(title_text="PC2")
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def _enabled(enabled, name: str) -> bool:
    if enabled is None:
        return True
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"all", name}
    if isinstance(enabled, (list, tuple, set)):
        vals = {str(v).strip().lower() for v in enabled}
        return "all" in vals or name in vals
    return bool(enabled)


def _numeric_series(df, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=np.float64)
    vals = df[col].to_numpy(dtype=np.float64, copy=False)
    return vals[np.isfinite(vals)]


def _scale(vals, lo, hi, out_lo, out_hi):
    vals = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(vals, (out_lo + out_hi) / 2.0, dtype=np.float64)
    return out_lo + (vals - lo) * (out_hi - out_lo) / (hi - lo)


def _svg_page(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#222">{escape(title)}</text>',
            *body,
            "</svg>",
        ]
    )


def _write_svg(path: Path, width: int, height: int, title: str, body: list[str]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_page(width, height, title, body), encoding="utf-8")
    return str(path)


def _empty_svg(path: Path, title: str, message: str = "no data") -> str:
    body = [f'<text x="400" y="260" text-anchor="middle" font-family="Arial" font-size="15" fill="#666">{escape(message)}</text>']
    return _write_svg(path, 800, 520, title, body)


def _axes(x0, y0, w, h, x_label="", y_label=""):
    return [
        f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{x0 + w / 2}" y="{y0 + h + 38}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{escape(x_label)}</text>',
        f'<text x="{x0 - 42}" y="{y0 + h / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333" transform="rotate(-90 {x0 - 42} {y0 + h / 2})">{escape(y_label)}</text>',
    ]


def _hist_svg(path: Path, vals: np.ndarray, title: str, x_label: str, bins: int = 60, color: str = "#4C78A8") -> str:
    if vals.size == 0:
        return _empty_svg(path, title)
    counts, edges = np.histogram(vals, bins=min(bins, max(1, vals.size)))
    width, height = 820, 540
    x0, y0, w, h = 80, 60, 700, 390
    max_count = max(int(counts.max()), 1)
    bw = w / len(counts)
    body = _axes(x0, y0, w, h, x_label, "count")
    for i, c in enumerate(counts):
        bh = h * (float(c) / max_count)
        x = x0 + i * bw
        y = y0 + h - bh
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="{color}" opacity="0.9"/>')
    body.extend(
        [
            f'<text x="{x0}" y="{y0 + h + 18}" font-family="Arial" font-size="11" fill="#555">{edges[0]:.4g}</text>',
            f'<text x="{x0 + w}" y="{y0 + h + 18}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{edges[-1]:.4g}</text>',
            f'<text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{max_count}</text>',
        ]
    )
    return _write_svg(path, width, height, title, body)


def _scatter_svg(path: Path, x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return _empty_svg(path, title)
    x = x[mask]
    y = y[mask]
    if x.size > 5000:
        idx = np.linspace(0, x.size - 1, 5000).astype(int)
        x = x[idx]
        y = y[idx]
    width, height = 820, 560
    x0, y0, w, h = 85, 60, 700, 400
    px = _scale(x, np.nanmin(x), np.nanmax(x), x0, x0 + w)
    py = _scale(y, np.nanmin(y), np.nanmax(y), y0 + h, y0)
    body = _axes(x0, y0, w, h, x_label, y_label)
    for xi, yi in zip(px, py):
        body.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="2.2" fill="#4C78A8" opacity="0.35"/>')
    return _write_svg(path, width, height, title, body)


def _box_svg(path: Path, groups: list[tuple[str, np.ndarray]], title: str, y_label: str) -> str:
    groups = [(label, vals[np.isfinite(vals)]) for label, vals in groups if vals.size]
    groups = [(label, vals) for label, vals in groups if vals.size]
    if not groups:
        return _empty_svg(path, title)
    width = max(900, min(1800, 90 * len(groups) + 140))
    height = 620
    x0, y0, w, h = 85, 60, width - 130, 420
    all_vals = np.concatenate([vals for _, vals in groups])
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    body = _axes(x0, y0, w, h, "class", y_label)
    step = w / max(len(groups), 1)
    for i, (label, vals) in enumerate(groups):
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        cx = x0 + step * (i + 0.5)
        yq1, ymed, yq3, ylo, yhi = _scale([q1, med, q3, vmin, vmax], y_min, y_max, y0 + h, y0)
        box_w = min(38, step * 0.55)
        body.extend(
            [
                f'<line x1="{cx:.2f}" y1="{yhi:.2f}" x2="{cx:.2f}" y2="{ylo:.2f}" stroke="#333"/>',
                f'<rect x="{cx - box_w / 2:.2f}" y="{min(yq1, yq3):.2f}" width="{box_w:.2f}" height="{abs(yq3 - yq1):.2f}" fill="#4C78A8" opacity="0.45" stroke="#333"/>',
                f'<line x1="{cx - box_w / 2:.2f}" y1="{ymed:.2f}" x2="{cx + box_w / 2:.2f}" y2="{ymed:.2f}" stroke="#111" stroke-width="2"/>',
                f'<text x="{cx:.2f}" y="{y0 + h + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#333" transform="rotate(-60 {cx:.2f} {y0 + h + 18})">{escape(label[:24])}</text>',
            ]
        )
    body.append(f'<text x="{x0 - 8}" y="{y0 + 5}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{y_max:.4g}</text>')
    body.append(f'<text x="{x0 - 8}" y="{y0 + h}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{y_min:.4g}</text>')
    return _write_svg(path, width, height, title, body)


def _heatmap_svg(path: Path, x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str, bins: int = 60) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return _empty_svg(path, title)
    heat, xedges, yedges = np.histogram2d(x[mask], y[mask], bins=bins)
    max_v = max(float(heat.max()), 1.0)
    width, height = 760, 620
    x0, y0, w, h = 85, 60, 560, 460
    body = _axes(x0, y0, w, h, x_label, y_label)
    cw, ch = w / bins, h / bins
    for ix in range(bins):
        for iy in range(bins):
            v = heat[ix, iy]
            if v <= 0:
                continue
            t = float(v / max_v)
            color = f"rgb({int(35 + 220 * t)},{int(70 + 140 * t)},{int(130 + 40 * (1 - t))})"
            body.append(f'<rect x="{x0 + ix * cw:.2f}" y="{y0 + h - (iy + 1) * ch:.2f}" width="{cw + 0.2:.2f}" height="{ch + 0.2:.2f}" fill="{color}"/>')
    return _write_svg(path, width, height, title, body)


def _corr_svg(path: Path, corr, cols: list[str]) -> str:
    if len(cols) < 2:
        return _empty_svg(path, "numeric correlation")
    n = len(cols)
    cell = 34
    width = max(760, 180 + cell * n)
    height = max(760, 180 + cell * n)
    x0, y0 = 120, 70
    body = []
    vals = corr.to_numpy()
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if not np.isfinite(v):
                v = 0.0
            if v >= 0:
                color = f"rgb({int(255 - 90 * v)}, {int(255 - 150 * v)}, {int(255 - 190 * v)})"
            else:
                a = abs(v)
                color = f"rgb({int(255 - 190 * a)}, {int(255 - 130 * a)}, 255)"
            body.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell}" height="{cell}" fill="{color}" stroke="#eee"/>')
    for i, col in enumerate(cols):
        body.append(f'<text x="{x0 - 8}" y="{y0 + i * cell + 22}" text-anchor="end" font-family="Arial" font-size="10" fill="#333">{escape(col)}</text>')
        body.append(f'<text x="{x0 + i * cell + 12}" y="{y0 + n * cell + 14}" text-anchor="end" font-family="Arial" font-size="10" fill="#333" transform="rotate(-60 {x0 + i * cell + 12} {y0 + n * cell + 14})">{escape(col)}</text>')
    return _write_svg(path, width, height, "numeric correlation", body)


def save_prediction_distribution_plots(out_dir: Path, *, df, enabled=None, image_counts=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[str] | str] = {}

    hist_cols = ["obj", "cls_conf", "score", "w", "h", "area", "aspect_ratio", "max_iou", "tp"]
    if _enabled(enabled, "histogram"):
        paths = []
        for col in hist_cols:
            paths.append(_hist_svg(out_dir / "histogram" / f"{col}.svg", _numeric_series(df, col), f"Histogram - {col}", col, bins=80))
        outputs["histogram"] = paths

    if _enabled(enabled, "box_violin"):
        paths = []
        for metric in ["score", "obj", "cls_conf", "max_iou"]:
            if df.empty or metric not in df.columns or "pred_class" not in df.columns:
                paths.append(_empty_svg(out_dir / "box_violin" / f"{metric}_by_class.svg", f"{metric} by class"))
                continue
            class_counts = df["pred_class"].astype(str).value_counts().head(30)
            groups = []
            for cls_name in class_counts.index.tolist():
                groups.append((str(cls_name), df.loc[df["pred_class"].astype(str) == cls_name, metric].to_numpy(dtype=np.float64, copy=False)))
            paths.append(_box_svg(out_dir / "box_violin" / f"{metric}_by_class.svg", groups, f"{metric} by class", metric))
        outputs["box_violin"] = paths

    if _enabled(enabled, "scatter"):
        paths = []
        for x_col, y_col in [("obj", "cls_conf"), ("area", "score"), ("aspect_ratio", "score"), ("score", "max_iou")]:
            paths.append(_scatter_svg(out_dir / "scatter" / f"{x_col}_vs_{y_col}.svg", _numeric_series(df, x_col), _numeric_series(df, y_col), f"{x_col} vs {y_col}", x_col, y_col))
        outputs["scatter"] = paths

    if _enabled(enabled, "heatmap"):
        outputs["heatmap"] = _heatmap_svg(out_dir / "heatmap" / "bbox_center_density.svg", _numeric_series(df, "cx"), _numeric_series(df, "cy"), "bbox center density", "cx", "cy")

    if _enabled(enabled, "rank_curve"):
        out = out_dir / "rank_curve" / "score_rank_curve.svg"
        if df.empty or not {"source_csv", "image_id", "score"}.issubset(df.columns):
            outputs["rank_curve"] = _empty_svg(out, "score rank curve")
        else:
            lines = []
            max_rank, min_score, max_score = 1, np.inf, -np.inf
            for _key, group in df.groupby(["source_csv", "image_id"], dropna=False):
                scores = group["score"].to_numpy(dtype=np.float64, copy=False)
                scores = np.sort(scores[np.isfinite(scores)])[::-1]
                if scores.size:
                    lines.append(scores)
                    max_rank = max(max_rank, int(scores.size))
                    min_score = min(min_score, float(scores.min()))
                    max_score = max(max_score, float(scores.max()))
            if not lines:
                outputs["rank_curve"] = _empty_svg(out, "score rank curve")
            else:
                x0, y0, w, h = 80, 60, 700, 390
                body = _axes(x0, y0, w, h, "rank", "score")
                for scores in lines[:1000]:
                    xs = _scale(np.arange(1, scores.size + 1), 1, max_rank, x0, x0 + w)
                    ys = _scale(scores, min_score, max_score, y0 + h, y0)
                    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
                    body.append(f'<polyline points="{pts}" fill="none" stroke="#4C78A8" stroke-width="1" opacity="0.18"/>')
                outputs["rank_curve"] = _write_svg(out, 820, 540, "score rank curve per image", body)

    if _enabled(enabled, "count_distribution"):
        if image_counts is not None and "num_predictions" in image_counts.columns:
            counts = image_counts["num_predictions"].to_numpy(dtype=np.float64, copy=False)
            counts = counts[np.isfinite(counts)]
        elif not df.empty and {"source_csv", "image_id"}.issubset(df.columns):
            counts = df.groupby(["source_csv", "image_id"], dropna=False).size().to_numpy(dtype=np.float64)
        else:
            counts = np.asarray([], dtype=np.float64)
        outputs["count_distribution"] = _hist_svg(out_dir / "count_distribution" / "predictions_per_image.svg", counts, "predictions per image", "num_predictions", bins=60, color="#59A14F")

    if _enabled(enabled, "correlation"):
        numeric_cols = ["xmin", "ymin", "xmax", "ymax", "cx", "cy", "w", "h", "area", "aspect_ratio", "obj", "cls_conf", "score", "max_iou", "tp"]
        cols = [c for c in numeric_cols if c in df.columns]
        if df.empty or len(cols) < 2:
            outputs["correlation"] = _empty_svg(out_dir / "correlation" / "numeric_correlation.svg", "numeric correlation")
        else:
            outputs["correlation"] = _corr_svg(out_dir / "correlation" / "numeric_correlation.svg", df[cols].corr(numeric_only=True), cols)

    if _enabled(enabled, "overview"):
        out = out_dir / "overview.svg"
        panels = []
        for idx, col in enumerate(["score", "obj", "cls_conf", "max_iou"]):
            vals = _numeric_series(df, col)
            counts, edges = np.histogram(vals, bins=50) if vals.size else (np.zeros(1), np.asarray([0, 1]))
            px0 = 65 + (idx % 2) * 410
            py0 = 60 + (idx // 2) * 255
            pw, ph = 330, 165
            panels.append(f'<text x="{px0 + pw / 2}" y="{py0 - 10}" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">{escape(col)}</text>')
            panels.extend(_axes(px0, py0, pw, ph, col, "count"))
            max_count = max(float(np.max(counts)), 1.0)
            bw = pw / len(counts)
            for i, c in enumerate(counts):
                bh = ph * (float(c) / max_count)
                panels.append(f'<rect x="{px0 + i * bw:.2f}" y="{py0 + ph - bh:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="#4C78A8" opacity="0.9"/>')
        outputs["overview"] = _write_svg(out, 850, 600, "prediction distribution overview", panels)

    return outputs
