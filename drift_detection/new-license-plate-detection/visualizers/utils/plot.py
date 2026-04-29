from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


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
