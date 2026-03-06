import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Path to logits.npy",
    )
    args, _ = parser.parse_known_args()
    return args


def load_logits(path_str: str) -> np.ndarray:
    if not path_str:
        raise ValueError("`--logdir` is required. Example: streamlit run tools/app.py -- --logdir Experiments_output/<time>/stats/logits.npy")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"logits file not found: {path}")

    logits = np.load(path)
    if logits.ndim != 2:
        raise ValueError(f"logits.npy must be 2D (N, C), got shape={logits.shape}")
    if logits.shape[0] == 0:
        raise ValueError("logits.npy is empty (N=0).")

    return logits


def per_sample_stats(logit: np.ndarray):
    mean = np.mean(logit)
    std = np.std(logit)
    skew = stats.skew(logit)
    kurt = stats.kurtosis(logit)

    sorted_vals = np.sort(logit)
    if sorted_vals.shape[0] >= 2:
        margin = sorted_vals[-1] - sorted_vals[-2]
    else:
        margin = np.nan

    exp_logit = np.exp(logit - np.max(logit))
    probs = exp_logit / np.sum(exp_logit)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return mean, std, skew, kurt, margin, entropy


def main():
    st.set_page_config(page_title="Logit Analysis Dashboard", layout="wide")
    st.title("Logit Analysis Dashboard")

    args = parse_args()
    try:
        logits = load_logits(args.logdir)
    except Exception as e:
        st.error(str(e))
        st.stop()

    n, c = logits.shape
    st.caption(f"Loaded logits: N={n}, C={c} from `{args.logdir}`")

    idx = st.slider("Select Sample Index", 0, n - 1, 0)
    logit = logits[idx]

    mean, std, skew, kurt, margin, entropy = per_sample_stats(logit)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Sample")
        st.write(f"Sample index: {idx}")
        st.write(f"Top-1 class index: {int(np.argmax(logit))}")
        st.write(f"Top-1 logit: {float(np.max(logit)):.4f}")
    with col2:
        st.subheader("Statistics")
        st.write(f"Mean: {mean:.4f}")
        st.write(f"Std: {std:.4f}")
        st.write(f"Skewness: {skew:.4f}")
        st.write(f"Kurtosis: {kurt:.4f}")
        st.write(f"Top1-Top2 Margin: {margin:.4f}")
        st.write(f"Softmax Entropy: {entropy:.4f}")

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Logit Distribution")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.hist(logit, bins=20)
        ax1.set_xlabel("Logit Value")
        ax1.set_ylabel("Count")
        ax1.set_title("Per-sample Histogram")
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with cols[1]:
        st.subheader("Class Logits")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(np.arange(c), logit)
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Logit")
        ax2.set_title("Per-sample Class Logits")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with cols[2]:
        st.subheader("Global Logit Distribution")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.hist(logits.flatten(), bins=50)
        ax3.set_xlabel("Logit Value")
        ax3.set_ylabel("Count")
        ax3.set_title("All Samples")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
