from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUP_PALETTE = {
    "noise": "#888888",
    "fn": "#E45756",
    "non_fn": "#4C78A8",
}


def save_pca_plot(
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

    with sns.axes_style("whitegrid"):
        if int(dimension) == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        if points is None or points.size == 0:
            msg = "insufficient data" if not reason else f"insufficient data\n{reason}"
            if int(dimension) == 3:
                ax.text2D(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, msg, ha="center", va="center")
                ax.set_axis_off()
        else:
            g_arr = np.asarray(groups, dtype=object)
            unique = [g for g in ["noise", "fn", "non_fn"] if g in set(g_arr.tolist())]
            if not unique:
                unique = sorted(set(g_arr.tolist()))
            if int(dimension) == 3:
                for g in unique:
                    mask = (g_arr == g)
                    p = points[mask]
                    if p.size == 0:
                        continue
                    ax.scatter(
                        p[:, 0],
                        p[:, 1],
                        p[:, 2],
                        label=str(g),
                        color=GROUP_PALETTE.get(g, "#333333"),
                        s=26,
                        alpha=0.85,
                        edgecolors="black",
                        linewidths=0.3,
                    )
                ax.set_zlabel("PC3")
            else:
                sns.scatterplot(
                    x=points[:, 0],
                    y=points[:, 1],
                    hue=g_arr,
                    hue_order=unique,
                    palette={k: GROUP_PALETTE.get(k, "#333333") for k in unique},
                    edgecolor="black",
                    linewidth=0.5,
                    s=40,
                    alpha=0.85,
                    ax=ax,
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(title="group", loc="best")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
