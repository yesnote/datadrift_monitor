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


def _bar_svg(path: Path, labels: list[str], values: np.ndarray, title: str, y_label: str, color: str = "#4C78A8", y_min: float | None = None, y_max: float | None = None) -> str:
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    labels = [label for label, keep in zip(labels, mask) if keep]
    vals = vals[mask]
    if vals.size == 0:
        return _empty_svg(path, title)
    width = max(900, min(1900, 72 * len(labels) + 160))
    height = 620
    x0, y0, w, h = 85, 60, width - 140, 420
    lo = float(np.nanmin(vals)) if y_min is None else float(y_min)
    hi = float(np.nanmax(vals)) if y_max is None else float(y_max)
    if hi <= lo:
        hi = lo + 1.0
    body = _axes(x0, y0, w, h, "feature", y_label)
    step = w / max(1, len(vals))
    for i, (label, val) in enumerate(zip(labels, vals)):
        y = float(_scale([val], lo, hi, y0 + h, y0)[0])
        base = float(_scale([max(0.0, lo)], lo, hi, y0 + h, y0)[0])
        top = min(y, base)
        bh = abs(base - y)
        x = x0 + i * step + step * 0.15
        body.append(f'<rect x="{x:.2f}" y="{top:.2f}" width="{step * 0.7:.2f}" height="{bh:.2f}" fill="{color}" opacity="0.9"/>')
        label_y = top - 6 if top > y0 + 18 else top + 14
        body.append(f'<text x="{x + step * 0.35:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#222">{val:.3g}</text>')
        body.append(f'<text x="{x + step * 0.35:.2f}" y="{y0 + h + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#333" transform="rotate(-60 {x + step * 0.35:.2f} {y0 + h + 18})">{escape(label)}</text>')
    body.append(f'<text x="{x0 - 8}" y="{y0 + 5}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{hi:.4g}</text>')
    body.append(f'<text x="{x0 - 8}" y="{y0 + h}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{lo:.4g}</text>')
    return _write_svg(path, width, height, title, body)


def _grouped_bar_svg(path: Path, labels: list[str], left: np.ndarray, right: np.ndarray, title: str, y_label: str, left_name: str, right_name: str) -> str:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) | np.isfinite(right)
    labels = [label for label, keep in zip(labels, mask) if keep]
    left = left[mask]
    right = right[mask]
    if len(labels) == 0:
        return _empty_svg(path, title)
    vals = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    width = max(900, min(1900, 80 * len(labels) + 180))
    height = 650
    x0, y0, w, h = 85, 70, width - 150, 420
    lo = min(0.0, float(np.nanmin(vals))) if vals.size else 0.0
    hi = max(1.0, float(np.nanmax(vals))) if vals.size else 1.0
    body = _axes(x0, y0, w, h, "feature", y_label)
    step = w / max(1, len(labels))
    base = float(_scale([0.0], lo, hi, y0 + h, y0)[0])
    for i, label in enumerate(labels):
        group_x = x0 + i * step
        for j, (val, color) in enumerate([(left[i], "#4C78A8"), (right[i], "#E45756")]):
            if not np.isfinite(val):
                continue
            y = float(_scale([val], lo, hi, y0 + h, y0)[0])
            top = min(y, base)
            bh = abs(base - y)
            x = group_x + step * (0.18 + 0.32 * j)
            body.append(f'<rect x="{x:.2f}" y="{top:.2f}" width="{step * 0.26:.2f}" height="{bh:.2f}" fill="{color}" opacity="0.9"/>')
            label_y = top - 5 if top > y0 + 16 else top + 13
            body.append(f'<text x="{x + step * 0.13:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Arial" font-size="9" fill="#222">{val:.3g}</text>')
        body.append(f'<text x="{group_x + step * 0.5:.2f}" y="{y0 + h + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#333" transform="rotate(-60 {group_x + step * 0.5:.2f} {y0 + h + 18})">{escape(label)}</text>')
    body.extend(
        [
            f'<rect x="{x0 + 15}" y="{height - 60}" width="14" height="14" fill="#4C78A8"/><text x="{x0 + 36}" y="{height - 48}" font-family="Arial" font-size="12">{escape(left_name)}</text>',
            f'<rect x="{x0 + 115}" y="{height - 60}" width="14" height="14" fill="#E45756"/><text x="{x0 + 136}" y="{height - 48}" font-family="Arial" font-size="12">{escape(right_name)}</text>',
        ]
    )
    return _write_svg(path, width, height, title, body)


def _metric_text(metric_lookup: dict, feature: str) -> str:
    row = metric_lookup.get(feature, {})
    if row is None or (hasattr(row, "empty") and row.empty) or (isinstance(row, dict) and not row):
        return ""
    high = row.get("auroc_fp_high", np.nan)
    low = row.get("auroc_fp_low", np.nan)
    ap_high = row.get("auprc_fp_high", np.nan)
    ap_low = row.get("auprc_fp_low", np.nan)
    auroc = np.nanmax([high, low])
    ap = ap_low if np.isfinite(low) and np.isfinite(high) and low >= high else ap_high
    if not np.isfinite(ap):
        ap = np.nanmax([ap_high, ap_low])
    if not np.isfinite(auroc):
        return ""
    if np.isfinite(ap):
        return f"AUROC={auroc:.3f}, AP={ap:.3f}"
    return f"AUROC={auroc:.3f}"


def _tp_fp_hist_panel(body: list[str], df, feature: str, x0: float, y0: float, w: float, h: float, title: str, metric_lookup: dict | None = None) -> None:
    if df is None or df.empty or feature not in df.columns or "tp" not in df.columns:
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#777">no data</text>')
        return
    raw = df[feature].to_numpy(dtype=np.float64, copy=False)
    tp_raw = df["tp"].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(raw) & np.isfinite(tp_raw)
    if not np.any(finite):
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#777">no data</text>')
        return
    vals = raw[finite]
    tp = tp_raw[finite] == 1
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 36)
    tp_counts, _ = np.histogram(vals[tp], bins=bins)
    fp_counts, _ = np.histogram(vals[~tp], bins=bins)
    max_count = max(float(tp_counts.max()) if tp_counts.size else 0.0, float(fp_counts.max()) if fp_counts.size else 0.0, 1.0)
    bw = w / max(1, len(tp_counts))
    body.append(f'<text x="{x0 + w / 2}" y="{y0 - 12}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">{escape(title)}</text>')
    metric = _metric_text(metric_lookup or {}, feature)
    if metric:
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + 8}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{escape(metric)}</text>')
    body.extend(_axes(x0, y0, w, h, feature, "count"))
    for i, count in enumerate(tp_counts):
        bh = h * (float(count) / max_count)
        body.append(f'<rect x="{x0 + i * bw:.2f}" y="{y0 + h - bh:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="#4C78A8" opacity="0.45"/>')
    for i, count in enumerate(fp_counts):
        bh = h * (float(count) / max_count)
        body.append(f'<rect x="{x0 + i * bw:.2f}" y="{y0 + h - bh:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="#E45756" opacity="0.45"/>')
    body.append(f'<text x="{x0}" y="{y0 + h + 15}" font-family="Arial" font-size="9" fill="#555">{lo:.3g}</text>')
    body.append(f'<text x="{x0 + w}" y="{y0 + h + 15}" text-anchor="end" font-family="Arial" font-size="9" fill="#555">{hi:.3g}</text>')


def _tp_fp_class_box_panel(body: list[str], df, feature: str, x0: float, y0: float, w: float, h: float, title: str, max_classes: int = 8, metric_lookup: dict | None = None) -> None:
    if df is None or df.empty or feature not in df.columns or "tp" not in df.columns or "pred_class" not in df.columns:
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#777">no data</text>')
        return
    class_counts = df["pred_class"].astype(str).value_counts().head(max_classes)
    groups = []
    all_vals = []
    for cls_name in class_counts.index.tolist():
        cls_mask = df["pred_class"].astype(str) == cls_name
        tp_vals = df.loc[cls_mask & (df["tp"] == 1), feature].to_numpy(dtype=np.float64, copy=False)
        fp_vals = df.loc[cls_mask & (df["tp"] == 0), feature].to_numpy(dtype=np.float64, copy=False)
        tp_vals = tp_vals[np.isfinite(tp_vals)]
        fp_vals = fp_vals[np.isfinite(fp_vals)]
        if tp_vals.size or fp_vals.size:
            groups.append((str(cls_name), tp_vals, fp_vals))
            if tp_vals.size:
                all_vals.append(tp_vals)
            if fp_vals.size:
                all_vals.append(fp_vals)
    if not groups or not all_vals:
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#777">no data</text>')
        return
    vals = np.concatenate(all_vals)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0
    body.append(f'<text x="{x0 + w / 2}" y="{y0 - 12}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">{escape(title)}</text>')
    metric = _metric_text(metric_lookup or {}, feature)
    if metric:
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + 8}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{escape(metric)}</text>')
    body.extend(_axes(x0, y0, w, h, "class", feature))
    step = w / max(1, len(groups))

    def draw_box(cx: float, vals: np.ndarray, color: str) -> None:
        if vals.size == 0:
            return
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        yq1, ymed, yq3, ylo, yhi = _scale([q1, med, q3, vmin, vmax], lo, hi, y0 + h, y0)
        box_w = min(12, step * 0.18)
        body.extend(
            [
                f'<line x1="{cx:.2f}" y1="{yhi:.2f}" x2="{cx:.2f}" y2="{ylo:.2f}" stroke="#333"/>',
                f'<rect x="{cx - box_w / 2:.2f}" y="{min(yq1, yq3):.2f}" width="{box_w:.2f}" height="{abs(yq3 - yq1):.2f}" fill="{color}" opacity="0.5" stroke="#333"/>',
                f'<line x1="{cx - box_w / 2:.2f}" y1="{ymed:.2f}" x2="{cx + box_w / 2:.2f}" y2="{ymed:.2f}" stroke="#111" stroke-width="1.5"/>',
            ]
        )

    for i, (cls_name, tp_vals, fp_vals) in enumerate(groups):
        center = x0 + step * (i + 0.5)
        draw_box(center - step * 0.12, tp_vals, "#4C78A8")
        draw_box(center + step * 0.12, fp_vals, "#E45756")
        body.append(f'<text x="{center:.2f}" y="{y0 + h + 15}" text-anchor="end" font-family="Arial" font-size="8.5" fill="#333" transform="rotate(-55 {center:.2f} {y0 + h + 15})">{escape(cls_name[:14])}</text>')


def _tp_fp_null_comparison_svg(path: Path, df, metrics_df=None) -> str:
    width, height = 1720, 1075
    margin_x, margin_y = 80, 90
    panel_w, panel_h = 330, 205
    gap_x, gap_y = 78, 110
    body = [
        '<rect x="40" y="44" width="14" height="14" fill="#4C78A8" opacity="0.55"/>',
        '<text x="60" y="56" font-family="Arial" font-size="12" fill="#333">TP</text>',
        '<rect x="110" y="44" width="14" height="14" fill="#E45756" opacity="0.55"/>',
        '<text x="130" y="56" font-family="Arial" font-size="12" fill="#333">FP</text>',
    ]
    panels = [
        ("hist", "score", "score"),
        ("hist", "obj", "obj"),
        ("class", "cls_conf", "cls conf by class"),
        ("hist", "area", "bbox area"),
        ("hist", "score_null_diff", "score null diff"),
        ("hist", "obj_null_bce_loss", "obj null BCE loss"),
        ("class", "cls_uniform_kl", "cls uniform kl by class"),
        ("hist", "bbox_anchor_log_area_ratio", "bbox anchor log area ratio"),
        ("hist", "score_cand_diff", "score cand diff"),
        ("hist", "obj_cand_bce_loss", "obj cand BCE loss"),
        ("class", "cls_cand_kl", "cls cand KL by class"),
        ("hist", "bbox_cand_log_area_ratio", "bbox cand log area ratio"),
    ]
    metric_lookup = {}
    if metrics_df is not None and not metrics_df.empty and "feature" in metrics_df.columns:
        metric_lookup = {str(row["feature"]): row for _, row in metrics_df.iterrows()}
    for idx, (kind, feature, title) in enumerate(panels):
        row, col = divmod(idx, 4)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        if kind == "class":
            _tp_fp_class_box_panel(body, df, feature, x0, y0, panel_w, panel_h, title, metric_lookup=metric_lookup)
        else:
            _tp_fp_hist_panel(body, df, feature, x0, y0, panel_w, panel_h, title, metric_lookup=metric_lookup)
    return _write_svg(path, width, height, "TP/FP Raw vs Null/Cand Feature Comparison", body)


def _tp_fp_hist_grid_svg(path: Path, df, features: list[str], metrics_df=None) -> str:
    if df is None or df.empty or "tp" not in df.columns:
        return _empty_svg(path, "TP/FP feature histograms")
    usable = [f for f in features if f in df.columns]
    if not usable:
        return _empty_svg(path, "TP/FP feature histograms")
    cols = 3
    rows = int(np.ceil(len(usable) / cols))
    panel_w, panel_h = 360, 245
    width, height = cols * panel_w + 40, rows * panel_h + 65
    body = [
        '<rect x="28" y="38" width="14" height="14" fill="#4C78A8" opacity="0.55"/>',
        '<text x="48" y="50" font-family="Arial" font-size="12" fill="#333">TP</text>',
        '<rect x="88" y="38" width="14" height="14" fill="#E45756" opacity="0.55"/>',
        '<text x="108" y="50" font-family="Arial" font-size="12" fill="#333">FP</text>',
    ]
    metric_lookup = {}
    if metrics_df is not None and not metrics_df.empty and "feature" in metrics_df.columns:
        metric_lookup = {str(row["feature"]): row for _, row in metrics_df.iterrows()}
    tp_raw = df["tp"].to_numpy(dtype=np.float64, copy=False)
    for idx, feature in enumerate(usable):
        r, c = divmod(idx, cols)
        px0 = 62 + c * panel_w
        py0 = 82 + r * panel_h
        pw, ph = 260, 135
        raw = df[feature].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(raw) & np.isfinite(tp_raw)
        if not np.any(finite):
            continue
        vals = raw[finite]
        tp = tp_raw[finite] == 1
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 31)
        tp_counts, _ = np.histogram(vals[tp], bins=bins)
        fp_counts, _ = np.histogram(vals[~tp], bins=bins)
        max_count = max(float(tp_counts.max()) if tp_counts.size else 0.0, float(fp_counts.max()) if fp_counts.size else 0.0, 1.0)
        bw = pw / max(1, len(tp_counts))
        body.append(f'<text x="{px0 + pw / 2}" y="{py0 - 10}" text-anchor="middle" font-family="Arial" font-size="12" fill="#222">{escape(feature)}</text>')
        metric = _metric_text(metric_lookup, feature)
        if metric:
            body.append(f'<text x="{px0 + pw / 2}" y="{py0 + 8}" text-anchor="middle" font-family="Arial" font-size="9" fill="#555">{escape(metric)}</text>')
        body.extend(_axes(px0, py0, pw, ph, "", ""))
        for i, count in enumerate(tp_counts):
            bh = ph * (float(count) / max_count)
            body.append(f'<rect x="{px0 + i * bw:.2f}" y="{py0 + ph - bh:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="#4C78A8" opacity="0.45"/>')
        for i, count in enumerate(fp_counts):
            bh = ph * (float(count) / max_count)
            body.append(f'<rect x="{px0 + i * bw:.2f}" y="{py0 + ph - bh:.2f}" width="{max(1, bw - 1):.2f}" height="{bh:.2f}" fill="#E45756" opacity="0.45"/>')
        body.append(f'<text x="{px0}" y="{py0 + ph + 14}" font-family="Arial" font-size="9" fill="#555">{lo:.3g}</text>')
        body.append(f'<text x="{px0 + pw}" y="{py0 + ph + 14}" text-anchor="end" font-family="Arial" font-size="9" fill="#555">{hi:.3g}</text>')
    return _write_svg(path, width, height, "TP/FP feature histograms", body)


def _candidate_target_summary_svg(path: Path, df, metrics_df=None) -> str:
    width, height = 1320, 760
    margin_x, margin_y = 70, 95
    panel_w, panel_h = 335, 165
    gap_x, gap_y = 75, 105
    features = [
        ("num_cand_boxes", "candidate boxes used"),
        ("num_nonself_cand_boxes", "non-self candidate boxes"),
        ("cand_score_mean", "candidate score mean"),
        ("cand_iou_mean", "candidate IoU mean"),
        ("cand_area_mean", "candidate area mean"),
        ("bbox_cand_log_area_ratio_std", "candidate log area ratio std"),
    ]
    body = [
        '<rect x="38" y="45" width="14" height="14" fill="#4C78A8" opacity="0.55"/>',
        '<text x="58" y="57" font-family="Arial" font-size="12" fill="#333">TP</text>',
        '<rect x="98" y="45" width="14" height="14" fill="#E45756" opacity="0.55"/>',
        '<text x="118" y="57" font-family="Arial" font-size="12" fill="#333">FP</text>',
    ]
    metric_lookup = {}
    if metrics_df is not None and not metrics_df.empty and "feature" in metrics_df.columns:
        metric_lookup = {str(row["feature"]): row for _, row in metrics_df.iterrows()}

    if df is not None and not df.empty:
        for text_idx, col in enumerate(["num_cand_boxes", "num_nonself_cand_boxes"]):
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=np.float64, copy=False)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                x = 250 + text_idx * 245
                body.append(
                    f'<text x="{x}" y="57" font-family="Arial" font-size="12" fill="#333">'
                    f'{escape(col)} mean={float(np.mean(vals)):.3g}, p50={float(np.median(vals)):.3g}</text>'
                )

    for idx, (feature, title) in enumerate(features):
        row, col = divmod(idx, 3)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        _tp_fp_hist_panel(body, df, feature, x0, y0, panel_w, panel_h, title, metric_lookup=metric_lookup)
    return _write_svg(path, width, height, "Candidate Target Summary", body)


def save_uncertainty_analysis_plots(out_dir: Path, *, pred_df=None, features=None, tp_fp_df, metrics_df, high_score_df) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if metrics_df is not None and not metrics_df.empty and "feature" in metrics_df.columns:
        labels = metrics_df["feature"].astype(str).tolist()
        if {"auroc_fp_high", "auroc_fp_low"}.issubset(metrics_df.columns):
            high = metrics_df["auroc_fp_high"].to_numpy(dtype=np.float64, copy=False)
            low = metrics_df["auroc_fp_low"].to_numpy(dtype=np.float64, copy=False)
            best = np.nanmax(np.vstack([high, low]), axis=0)
            outputs["best_fp_auroc"] = _bar_svg(out_dir / "best_fp_auroc.svg", labels, best, "Best FP AUROC by feature", "AUROC", y_min=0.0, y_max=1.0)
            outputs["fp_auroc_direction"] = _grouped_bar_svg(out_dir / "fp_auroc_direction.svg", labels, high, low, "FP AUROC by direction", "AUROC", "high=FP", "low=FP")
        if "corr_with_max_iou" in metrics_df.columns:
            outputs["corr_with_max_iou"] = _bar_svg(out_dir / "corr_with_max_iou.svg", labels, metrics_df["corr_with_max_iou"].to_numpy(dtype=np.float64, copy=False), "Correlation with max IoU", "Pearson r", y_min=-1.0, y_max=1.0, color="#59A14F")
    else:
        outputs["best_fp_auroc"] = _empty_svg(out_dir / "best_fp_auroc.svg", "Best FP AUROC by feature")

    if tp_fp_df is not None and not tp_fp_df.empty and {"feature", "tp_mean", "fp_mean"}.issubset(tp_fp_df.columns):
        labels = tp_fp_df["feature"].astype(str).tolist()
        outputs["tp_fp_mean"] = _grouped_bar_svg(
            out_dir / "tp_fp_mean.svg",
            labels,
            tp_fp_df["tp_mean"].to_numpy(dtype=np.float64, copy=False),
            tp_fp_df["fp_mean"].to_numpy(dtype=np.float64, copy=False),
            "TP vs FP feature means",
            "mean value",
            "TP",
            "FP",
        )
        if "cohen_d_fp_minus_tp" in tp_fp_df.columns:
            outputs["effect_size"] = _bar_svg(
                out_dir / "effect_size_fp_minus_tp.svg",
                labels,
                tp_fp_df["cohen_d_fp_minus_tp"].to_numpy(dtype=np.float64, copy=False),
                "Effect size: FP minus TP",
                "Cohen d",
                color="#F28E2B",
            )

    if high_score_df is not None and not high_score_df.empty and {"score_threshold", "fp_rate"}.issubset(high_score_df.columns):
        labels = [f">={v:g}" for v in high_score_df["score_threshold"].to_numpy(dtype=np.float64, copy=False)]
        outputs["high_score_fp_rate"] = _bar_svg(
            out_dir / "high_score_fp_rate.svg",
            labels,
            high_score_df["fp_rate"].to_numpy(dtype=np.float64, copy=False),
            "FP rate among high-score predictions",
            "FP rate",
            color="#E45756",
            y_min=0.0,
            y_max=1.0,
        )

    if pred_df is not None and features is not None:
        outputs["tp_fp_histograms"] = _tp_fp_hist_grid_svg(out_dir / "tp_fp_feature_histograms.svg", pred_df, list(features), metrics_df=metrics_df)
        outputs["tp_fp_null_cand_comparison"] = _tp_fp_null_comparison_svg(out_dir / "tp_fp_null_cand_comparison_grid.svg", pred_df, metrics_df=metrics_df)
        outputs["candidate_target_summary"] = _candidate_target_summary_svg(out_dir / "candidate_target_summary.svg", pred_df, metrics_df=metrics_df)

    return outputs


def save_prediction_distribution_plots(out_dir: Path, *, df, enabled=None, image_counts=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[str] | str] = {}

    hist_cols = [
        "obj",
        "cls_conf",
        "score",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "anchor_w",
        "anchor_h",
        "anchor_area",
        "bbox_anchor_center_l2",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_ciou_loss",
        "bbox_anchor_aspect_ratio_diff",
        "obj_null_abs_diff",
        "obj_null_bce_loss",
        "cls_null_abs_diff",
        "score_null_diff",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "score_cand_diff",
        "obj_cand_bce_loss",
        "cls_cand_kl",
        "bbox_cand_log_area_ratio",
        "bbox_cand_log_area_ratio_std",
        "num_cand_boxes",
        "num_nonself_cand_boxes",
        "has_nonself_cand",
        "cand_score_mean",
        "cand_score_min",
        "cand_score_max",
        "cand_score_std",
        "cand_iou_mean",
        "cand_iou_min",
        "cand_iou_max",
        "cand_iou_std",
        "cand_area_mean",
        "cand_area_min",
        "cand_area_max",
        "cand_area_std",
        "max_iou",
        "tp",
    ]
    if _enabled(enabled, "histogram"):
        paths = []
        for col in hist_cols:
            paths.append(_hist_svg(out_dir / "histogram" / f"{col}.svg", _numeric_series(df, col), f"Histogram - {col}", col, bins=80))
        outputs["histogram"] = paths

    if _enabled(enabled, "box_violin"):
        paths = []
        for metric in ["score", "obj", "cls_conf", "score_null_diff", "cls_uniform_kl", "max_iou"]:
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
        for x_col, y_col in [
            ("obj", "cls_conf"),
            ("area", "score"),
            ("aspect_ratio", "score"),
            ("bbox_anchor_center_l2", "score"),
            ("bbox_anchor_log_area_ratio", "score"),
            ("score", "max_iou"),
            ("score_null_diff", "max_iou"),
            ("num_cand_boxes", "score"),
            ("num_cand_boxes", "max_iou"),
            ("cand_iou_mean", "max_iou"),
            ("cand_score_mean", "score"),
            ("cls_entropy_norm", "score"),
        ]:
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
        numeric_cols = [
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "cx",
            "cy",
            "w",
            "h",
            "area",
            "aspect_ratio",
            "anchor_w",
            "anchor_h",
            "anchor_area",
            "bbox_anchor_center_l2",
            "bbox_anchor_log_w_ratio",
            "bbox_anchor_log_h_ratio",
            "bbox_anchor_log_area_ratio",
            "bbox_anchor_aspect_ratio_diff",
            "obj",
            "cls_conf",
            "score",
            "obj_null_abs_diff",
            "obj_null_bce_loss",
            "cls_null_abs_diff",
            "score_null_diff",
            "cls_entropy_norm",
            "cls_uniform_kl",
            "score_cand_diff",
            "obj_cand_bce_loss",
            "cls_cand_kl",
            "bbox_cand_log_area_ratio",
            "bbox_cand_log_area_ratio_std",
            "num_cand_boxes",
            "num_nonself_cand_boxes",
            "has_nonself_cand",
            "cand_score_mean",
            "cand_score_min",
            "cand_score_max",
            "cand_score_std",
            "cand_iou_mean",
            "cand_iou_min",
            "cand_iou_max",
            "cand_iou_std",
            "cand_area_mean",
            "cand_area_min",
            "cand_area_max",
            "cand_area_std",
            "max_iou",
            "tp",
        ]
        cols = [c for c in numeric_cols if c in df.columns]
        if df.empty or len(cols) < 2:
            outputs["correlation"] = _empty_svg(out_dir / "correlation" / "numeric_correlation.svg", "numeric correlation")
        else:
            outputs["correlation"] = _corr_svg(out_dir / "correlation" / "numeric_correlation.svg", df[cols].corr(numeric_only=True), cols)

    if _enabled(enabled, "overview"):
        out = out_dir / "overview.svg"
        panels = []
        for idx, col in enumerate(["score", "bbox_anchor_log_area_ratio", "bbox_anchor_center_l2", "max_iou"]):
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
