from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.update_layout(
        title=title,
        legend_title_text="group",
        template="plotly_white",
    )

    if points is None or points.size == 0:
        msg = "insufficient data" if not reason else f"insufficient data: {reason}"
        fig.add_annotation(
            text=msg,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
        )
    else:
        g_arr = np.asarray(groups, dtype=object)
        unique = [g for g in ["noise", "fn", "non_fn"] if g in set(g_arr.tolist())]
        if not unique:
            unique = sorted(set(g_arr.tolist()))

        for g in unique:
            mask = g_arr == g
            p = points[mask]
            if p.size == 0:
                continue
            color = GROUP_PALETTE.get(g, "#333333")
            if int(dimension) == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=p[:, 0],
                        y=p[:, 1],
                        z=p[:, 2],
                        mode="markers",
                        marker=dict(size=4, color=color, opacity=0.85),
                        name=str(g),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=p[:, 0],
                        y=p[:, 1],
                        mode="markers",
                        marker=dict(size=7, color=color, opacity=0.85, line=dict(width=0.5, color="black")),
                        name=str(g),
                    )
                )

    if int(dimension) == 3:
        fig.update_layout(
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            )
        )
    else:
        fig.update_xaxes(title_text="PC1")
        fig.update_yaxes(title_text="PC2")

    # plotly legend supports group on/off via click by default.
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


def _write_empty_plot(out_path: Path, title: str, message: str = "no data") -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)


def _numeric_series(df, col: str):
    if col not in df.columns:
        return np.asarray([], dtype=np.float32)
    vals = df[col].to_numpy(dtype=np.float64, copy=False)
    return vals[np.isfinite(vals)]


def save_prediction_distribution_plots(out_dir: Path, *, df, enabled=None, image_counts=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[str] | str] = {}

    hist_cols = ["obj", "cls_conf", "score", "w", "h", "area", "aspect_ratio", "max_iou", "tp"]
    if _enabled(enabled, "histogram"):
        paths = []
        for col in hist_cols:
            out = out_dir / "histogram" / f"{col}.html"
            vals = _numeric_series(df, col)
            if vals.size == 0:
                paths.append(_write_empty_plot(out, f"Histogram - {col}"))
                continue
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals, nbinsx=80, marker_color="#4C78A8", name=col))
            fig.update_layout(title=f"Histogram - {col}", template="plotly_white", xaxis_title=col, yaxis_title="count")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            paths.append(str(out))
        outputs["histogram"] = paths

    if _enabled(enabled, "box_violin"):
        paths = []
        for metric in ["score", "obj", "cls_conf", "max_iou"]:
            out = out_dir / "box_violin" / f"{metric}_by_class.html"
            if df.empty or metric not in df.columns or "pred_class" not in df.columns:
                paths.append(_write_empty_plot(out, f"{metric} by class"))
                continue
            class_counts = df["pred_class"].astype(str).value_counts().head(30)
            fig = go.Figure()
            for cls_name in class_counts.index.tolist():
                vals = df.loc[df["pred_class"].astype(str) == cls_name, metric].to_numpy(dtype=np.float64, copy=False)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                fig.add_trace(
                    go.Violin(
                        y=vals,
                        name=str(cls_name),
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                    )
                )
            fig.update_layout(title=f"{metric} by class", template="plotly_white", yaxis_title=metric)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            paths.append(str(out))
        outputs["box_violin"] = paths

    if _enabled(enabled, "scatter"):
        pairs = [("obj", "cls_conf"), ("area", "score"), ("aspect_ratio", "score"), ("score", "max_iou")]
        paths = []
        for x_col, y_col in pairs:
            out = out_dir / "scatter" / f"{x_col}_vs_{y_col}.html"
            if df.empty or x_col not in df.columns or y_col not in df.columns:
                paths.append(_write_empty_plot(out, f"{x_col} vs {y_col}"))
                continue
            x = df[x_col].to_numpy(dtype=np.float64, copy=False)
            y = df[y_col].to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            colors = df["pred_class"].astype(str).to_numpy() if "pred_class" in df.columns else None
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    marker=dict(size=5, opacity=0.55, color="#4C78A8"),
                    text=colors[mask] if colors is not None else None,
                    name=f"{x_col} vs {y_col}",
                )
            )
            fig.update_layout(title=f"{x_col} vs {y_col}", template="plotly_white", xaxis_title=x_col, yaxis_title=y_col)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            paths.append(str(out))
        outputs["scatter"] = paths

    if _enabled(enabled, "heatmap"):
        out = out_dir / "heatmap" / "bbox_center_density.html"
        if df.empty or "cx" not in df.columns or "cy" not in df.columns:
            outputs["heatmap"] = _write_empty_plot(out, "bbox center density")
        else:
            x = df["cx"].to_numpy(dtype=np.float64, copy=False)
            y = df["cy"].to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            fig = go.Figure(go.Histogram2d(x=x[mask], y=y[mask], nbinsx=80, nbinsy=80, colorscale="Viridis"))
            fig.update_layout(title="bbox center density", template="plotly_white", xaxis_title="cx", yaxis_title="cy")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            outputs["heatmap"] = str(out)

    if _enabled(enabled, "rank_curve"):
        out = out_dir / "rank_curve" / "score_rank_curve.html"
        if df.empty or not {"source_csv", "image_id", "score"}.issubset(df.columns):
            outputs["rank_curve"] = _write_empty_plot(out, "score rank curve")
        else:
            fig = go.Figure()
            for _key, group in df.groupby(["source_csv", "image_id"], dropna=False):
                scores = group["score"].to_numpy(dtype=np.float64, copy=False)
                scores = np.sort(scores[np.isfinite(scores)])[::-1]
                if scores.size:
                    fig.add_trace(go.Scatter(x=np.arange(1, scores.size + 1), y=scores, mode="lines", line=dict(width=1), opacity=0.18, showlegend=False))
            fig.update_layout(title="score rank curve per image", template="plotly_white", xaxis_title="rank", yaxis_title="score")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            outputs["rank_curve"] = str(out)

    if _enabled(enabled, "count_distribution"):
        out = out_dir / "count_distribution" / "predictions_per_image.html"
        if image_counts is not None and "num_predictions" in image_counts.columns:
            counts = image_counts["num_predictions"].to_numpy(dtype=np.float64, copy=False)
            counts = counts[np.isfinite(counts)]
        elif not df.empty and {"source_csv", "image_id"}.issubset(df.columns):
            counts = df.groupby(["source_csv", "image_id"], dropna=False).size().to_numpy(dtype=np.float64)
        else:
            counts = np.asarray([], dtype=np.float64)
        if counts.size == 0:
            outputs["count_distribution"] = _write_empty_plot(out, "predictions per image")
        else:
            fig = go.Figure(go.Histogram(x=counts, nbinsx=60, marker_color="#59A14F"))
            fig.update_layout(title="predictions per image", template="plotly_white", xaxis_title="num_predictions", yaxis_title="num_images")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            outputs["count_distribution"] = str(out)

    if _enabled(enabled, "correlation"):
        out = out_dir / "correlation" / "numeric_correlation.html"
        numeric_cols = ["xmin", "ymin", "xmax", "ymax", "cx", "cy", "w", "h", "area", "aspect_ratio", "obj", "cls_conf", "score", "max_iou", "tp"]
        cols = [c for c in numeric_cols if c in df.columns]
        if df.empty or len(cols) < 2:
            outputs["correlation"] = _write_empty_plot(out, "numeric correlation")
        else:
            corr = df[cols].corr(numeric_only=True).to_numpy()
            fig = go.Figure(go.Heatmap(z=corr, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu"))
            fig.update_layout(title="numeric correlation", template="plotly_white")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            outputs["correlation"] = str(out)

    if _enabled(enabled, "overview"):
        out = out_dir / "overview.html"
        fig = make_subplots(rows=2, cols=2, subplot_titles=("score", "obj", "cls_conf", "area"))
        for idx, col in enumerate(["score", "obj", "cls_conf", "max_iou"]):
            vals = _numeric_series(df, col)
            r = idx // 2 + 1
            c = idx % 2 + 1
            fig.add_trace(go.Histogram(x=vals, nbinsx=60, name=col, showlegend=False), row=r, col=c)
        fig.update_layout(title="prediction distribution overview", template="plotly_white")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        outputs["overview"] = str(out)

    return outputs
