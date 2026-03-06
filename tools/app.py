import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import streamlit as st

PLOT_TITLE_FSIZE = 5
PLOT_LABEL_FSIZE = 3
PLOT_TICK_FSIZE = 3


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Path to experiment directory (e.g. Experiments_output/<time>)",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_experiment_paths(path_str: str):
    if not path_str:
        raise ValueError(
            "`--logdir` is required. Example: streamlit run tools/app.py -- --logdir Experiments_output/<time>"
        )

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")

    # Accept both experiment root and direct logits path
    if path.is_file() and path.name == "logits.npy":
        exp_dir = path.parent.parent
        logits_path = path
    else:
        exp_dir = path
        logits_path = exp_dir / "stats" / "logits.npy"

    xai_dir = exp_dir / "XAI"
    if not logits_path.exists():
        raise FileNotFoundError(f"logits file not found: {logits_path}")
    if not xai_dir.exists():
        raise FileNotFoundError(f"XAI directory not found: {xai_dir}")

    return exp_dir, logits_path, xai_dir


def load_logits(logits_path: Path) -> np.ndarray:
    logits = np.load(logits_path)
    if logits.ndim != 2:
        raise ValueError(f"logits.npy must be 2D (N, C), got shape={logits.shape}")
    if logits.shape[0] == 0:
        raise ValueError("logits.npy is empty (N=0).")
    return logits


def load_xai_images(xai_dir: Path):
    image_paths = list(xai_dir.glob("*.jpg"))
    if len(image_paths) == 0:
        raise ValueError(f"No .jpg files found in: {xai_dir}")
    # Sort by modification time to keep generation order
    image_paths = sorted(image_paths, key=lambda p: (p.stat().st_mtime, p.name))
    return image_paths


def per_sample_stats(logit: np.ndarray):
    mean = np.mean(logit)
    std = np.std(logit)
    skew = stats.skew(logit)
    kurt = stats.kurtosis(logit)

    sorted_vals = np.sort(logit)
    margin = sorted_vals[-1] - sorted_vals[-2] if sorted_vals.shape[0] >= 2 else np.nan

    exp_logit = np.exp(logit - np.max(logit))
    probs = exp_logit / np.sum(exp_logit)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return mean, std, skew, kurt, margin, entropy


def main():
    st.set_page_config(page_title="XAI Logit Analysis Dashboard", layout="wide")
    st.title("XAI Logit Analysis Dashboard")

    args = parse_args()
    try:
        exp_dir, logits_path, xai_dir = resolve_experiment_paths(args.logdir)
        logits = load_logits(logits_path)
        xai_images = load_xai_images(xai_dir)
    except Exception as e:
        st.error(str(e))
        st.stop()

    n_logits, c = logits.shape
    n_images = len(xai_images)
    n = min(n_logits, n_images)
    if n_logits != n_images:
        st.warning(
            f"Count mismatch: logits rows={n_logits}, XAI images={n_images}. Showing first {n} pairs."
        )

    logits = logits[:n]
    xai_images = xai_images[:n]
    st.caption(
        f"Experiment: `{exp_dir}` | logits: `{logits_path}` | XAI: `{xai_dir}` | N={n}, C={c}"
    )

    idx = st.slider("Select Sample Index", 0, n - 1, 0)
    logit = logits[idx]
    xai_image_path = xai_images[idx]

    mean, std, skew, kurt, margin, entropy = per_sample_stats(logit)

    row1 = st.columns(3)
    with row1[0]:
        st.subheader("Image")
        st.image(str(xai_image_path), width=350)
    with row1[1]:
        st.subheader("Selected Sample")
        st.write(f"Sample index: {idx}")
        st.write(f"Image file: `{xai_image_path.name}`")
        st.write(f"Top-1 class index: {int(np.argmax(logit))}")
        st.write(f"Top-1 logit: {float(np.max(logit)):.4f}")
    with row1[2]:
        st.subheader("Statistics")
        st.write(f"Mean: {mean:.4f}")
        st.write(f"Std: {std:.4f}")
        st.write(f"Skewness: {skew:.4f}")
        st.write(f"Kurtosis: {kurt:.4f}")
        st.write(f"Top1-Top2 Margin: {margin:.4f}")
        st.write(f"Softmax Entropy: {entropy:.4f}")

    row2 = st.columns(3)
    with row2[0]:
        st.subheader("Logit Distribution")
        fig1, ax1 = plt.subplots(figsize=(2, 1.5))
        ax1.hist(logit, bins=20)
        ax1.set_xlabel("Logit Value", fontsize=PLOT_LABEL_FSIZE)
        ax1.set_ylabel("Count", fontsize=PLOT_LABEL_FSIZE)
        ax1.set_title("Per-sample Histogram", fontsize=PLOT_TITLE_FSIZE)
        ax1.tick_params(axis="both", labelsize=PLOT_TICK_FSIZE)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with row2[1]:
        st.subheader("Class Logits")
        fig2, ax2 = plt.subplots(figsize=(2, 1.5))
        ax2.bar(np.arange(c), logit)
        ax2.set_xlabel("Class", fontsize=PLOT_LABEL_FSIZE)
        ax2.set_ylabel("Logit", fontsize=PLOT_LABEL_FSIZE)
        ax2.set_title("Per-sample Class Logits", fontsize=PLOT_TITLE_FSIZE)
        ax2.tick_params(axis="both", labelsize=PLOT_TICK_FSIZE)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with row2[2]:
        st.subheader("Global Logit Distribution")
        fig3, ax3 = plt.subplots(figsize=(2, 1.5))
        ax3.hist(logits.flatten(), bins=50)
        ax3.set_xlabel("Logit Value", fontsize=PLOT_LABEL_FSIZE)
        ax3.set_ylabel("Count", fontsize=PLOT_LABEL_FSIZE)
        ax3.set_title("All Samples", fontsize=PLOT_TITLE_FSIZE)
        ax3.tick_params(axis="both", labelsize=PLOT_TICK_FSIZE)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
