from commands.predict.common import *
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def _build_layer_filter_map_from_grad_stats(grad_stats, target_values, target_layers, layer_param_shapes=None):
    layer_vectors = []
    for layer_name in target_layers:
        per_target = []
        max_len = 0
        expected_shape = None
        if layer_param_shapes is not None:
            expected_shape = layer_param_shapes.get(layer_name)
        for target_value in target_values:
            key = f"{target_value}_{layer_name}"
            raw_vec = _vector_from_grad_value(grad_stats.get(key, []))
            vec = raw_vec
            if expected_shape and raw_vec.size > 0:
                numel = int(np.prod(expected_shape))
                if raw_vec.size == numel:
                    reshaped = raw_vec.reshape(expected_shape)
                    if len(expected_shape) == 1:
                        vec = np.abs(reshaped).astype(np.float32, copy=False)
                    else:
                        first_dim = int(expected_shape[0])
                        vec = np.abs(reshaped).reshape(first_dim, -1).mean(axis=1).astype(np.float32, copy=False)
            per_target.append(vec)
            if vec.shape[0] > max_len:
                max_len = vec.shape[0]
        if max_len == 0:
            layer_vectors.append(np.zeros((0,), dtype=np.float32))
            continue
        mat = np.full((len(per_target), max_len), np.nan, dtype=np.float32)
        for i, vec in enumerate(per_target):
            if vec.shape[0] > 0:
                mat[i, : vec.shape[0]] = vec
        layer_vectors.append(np.nanmean(mat, axis=0))

    f_max = max((v.shape[0] for v in layer_vectors), default=0)
    out = np.full((len(target_layers), f_max), np.nan, dtype=np.float32)
    for li, vec in enumerate(layer_vectors):
        if vec.shape[0] > 0:
            out[li, : vec.shape[0]] = vec
    return out


def _build_layer_filter_map_by_target_from_grad_stats(grad_stats, target_values, target_layers, layer_param_shapes=None):
    out = {}
    for target_value in target_values:
        out[str(target_value)] = _build_layer_filter_map_from_grad_stats(
            grad_stats=grad_stats,
            target_values=[target_value],
            target_layers=target_layers,
            layer_param_shapes=layer_param_shapes,
        )
    return out


def _aggregate_target_maps(target_map_dict):
    if not target_map_dict:
        return np.zeros((0, 0), dtype=np.float32)
    maps = [m for m in target_map_dict.values() if isinstance(m, np.ndarray)]
    return _stack_nanmean_maps(maps)


def _normalize_layer_map(layer_map, mode="layer_minmax"):
    if mode == "none":
        return layer_map.astype(np.float32, copy=True)
    out = layer_map.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            continue
        vals = row[finite_mask]
        if mode == "layer_trimmed_minmax":
            vmin = float(np.percentile(vals, 1.0))
            vmax = float(np.percentile(vals, 99.0))
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        if vmax > vmin:
            normed = (vals - vmin) / (vmax - vmin)
            row[finite_mask] = np.clip(normed, 0.0, 1.0)
        else:
            row[finite_mask] = 0.0
    return out


def _stack_nanmean_maps(maps):
    if not maps:
        return np.zeros((0, 0), dtype=np.float32)
    l_max = max(m.shape[0] for m in maps)
    f_max = max(m.shape[1] for m in maps)
    arr = np.full((len(maps), l_max, f_max), np.nan, dtype=np.float32)
    for i, m in enumerate(maps):
        arr[i, : m.shape[0], : m.shape[1]] = m
    valid = np.isfinite(arr)
    count = valid.sum(axis=0).astype(np.float32)
    total = np.where(valid, arr, 0.0).sum(axis=0).astype(np.float32)
    out = np.full((l_max, f_max), np.nan, dtype=np.float32)
    mask = count > 0
    out[mask] = total[mask] / count[mask]
    return out


def _profile_stats_from_mean_map(mean_map):
    if mean_map.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    means, stds, idxs = [], [], []
    for layer_idx in range(int(mean_map.shape[0])):
        row = mean_map[layer_idx]
        vals = row[np.isfinite(row)]
        if vals.size == 0:
            continue
        idxs.append(layer_idx)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    if not idxs:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.asarray(means, dtype=np.float32),
        np.asarray(stds, dtype=np.float32),
        np.asarray(idxs, dtype=np.int64),
    )


def _save_layer_profile_plot(
    fn_mean_map,
    non_fn_mean_map,
    out_path,
    log_scale=True,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fn_mean, fn_std, fn_idx = _profile_stats_from_mean_map(fn_mean_map)
    non_mean, non_std, non_idx = _profile_stats_from_mean_map(non_fn_mean_map)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    eps = 1e-12
    plotted = False
    band_values = []
    if fn_idx.size > 0:
        y = np.maximum(fn_mean, eps) if log_scale else fn_mean
        lo = np.maximum(fn_mean - fn_std, eps) if log_scale else (fn_mean - fn_std)
        hi = np.maximum(fn_mean + fn_std, eps) if log_scale else (fn_mean + fn_std)
        band_values.append(lo)
        band_values.append(hi)
        ax.plot(fn_idx, y, color="#d62728", linewidth=2.0, label="FN mean")
        ax.fill_between(fn_idx, lo, hi, color="#d62728", alpha=0.18, linewidth=0)
        plotted = True
    if non_idx.size > 0:
        y = np.maximum(non_mean, eps) if log_scale else non_mean
        lo = np.maximum(non_mean - non_std, eps) if log_scale else (non_mean - non_std)
        hi = np.maximum(non_mean + non_std, eps) if log_scale else (non_mean + non_std)
        band_values.append(lo)
        band_values.append(hi)
        ax.plot(non_idx, y, color="#1f77b4", linewidth=2.0, label="non-FN mean")
        ax.fill_between(non_idx, lo, hi, color="#1f77b4", alpha=0.18, linewidth=0)
        plotted = True

    if log_scale:
        ax.set_yscale("log")
    if plotted and band_values:
        vals = np.concatenate([v.reshape(-1) for v in band_values]).astype(np.float32, copy=False)
        vals = vals[np.isfinite(vals)]
        if log_scale:
            vals = vals[vals > eps]
        if vals.size > 1:
            cur_lo, cur_hi = ax.get_ylim()
            robust_lo = float(np.percentile(vals, 1.0))
            robust_hi = float(np.percentile(vals, 99.0))
            if log_scale:
                robust_lo = max(robust_lo, eps)
                robust_hi = max(robust_hi, robust_lo * 1.01)
            if robust_hi > robust_lo:
                new_lo = max(cur_lo, robust_lo)
                new_hi = min(cur_hi, robust_hi)
                if new_hi > new_lo:
                    ax.set_ylim(new_lo, new_hi)
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Layer Sum(|grad|)")
    ax.set_title("Layer-wise Gradient Profile (mean ± std)")
    ax.grid(True, which="both", axis="both", alpha=0.2)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No profile data", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_heatmap_png(map_2d, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#efefef")

    if map_2d.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Filter Number")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    m = map_2d.astype(np.float32, copy=True)
    finite_mask = np.isfinite(m)
    if not finite_mask.any():
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Filter Number")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    vals = m[finite_mask]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax > vmin:
        m[finite_mask] = (vals - vmin) / (vmax - vmin)
    else:
        m[finite_mask] = 0.0

    layer_idx, filter_idx = np.where(finite_mask)
    color_vals = m[finite_mask]
    sc = ax.scatter(
        layer_idx.astype(np.float32),
        filter_idx.astype(np.float32),
        c=color_vals,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        s=90,
        alpha=0.32,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Gradient")
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Filter Number")
    ax.set_xlim(-0.5, m.shape[0] - 0.5)
    ax.set_ylim(-0.5, m.shape[1] - 0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _pad_map_2d(map_2d, shape):
    out = np.full(shape, np.nan, dtype=np.float32)
    if map_2d.size == 0:
        return out
    h = min(shape[0], map_2d.shape[0])
    w = min(shape[1], map_2d.shape[1])
    out[:h, :w] = map_2d[:h, :w]
    return out


def _pad_count_2d(count_2d, shape):
    out = np.zeros(shape, dtype=np.int32)
    if count_2d is None or count_2d.size == 0:
        return out
    h = min(shape[0], count_2d.shape[0])
    w = min(shape[1], count_2d.shape[1])
    out[:h, :w] = count_2d[:h, :w].astype(np.int32, copy=False)
    return out


def _merge_map_shape(shape_a, shape_b):
    if shape_a is None:
        return shape_b
    if shape_b is None:
        return shape_a
    return (max(int(shape_a[0]), int(shape_b[0])), max(int(shape_a[1]), int(shape_b[1])))


def _update_running_mean_map(state, sample_map):
    sample = sample_map.astype(np.float32, copy=True)
    target_shape = _merge_map_shape(state.get("shape"), sample.shape)
    if target_shape is None:
        target_shape = sample.shape
    sample = _pad_map_2d(sample, target_shape)
    if state.get("mean_raw") is None:
        mean_raw = np.full(target_shape, np.nan, dtype=np.float32)
        obs_count = np.zeros(target_shape, dtype=np.int32)
    else:
        mean_raw = _pad_map_2d(state["mean_raw"], target_shape)
        obs_count = _pad_count_2d(state.get("obs_count"), target_shape)

    prev_mean = mean_raw.copy()
    finite_mask = np.isfinite(sample)
    if finite_mask.any():
        c_old = obs_count[finite_mask].astype(np.float32)
        c_new = c_old + 1.0
        prev_vals = mean_raw[finite_mask]
        need_init = ~np.isfinite(prev_vals)
        if need_init.any():
            prev_vals[need_init] = sample[finite_mask][need_init]
        updated_vals = prev_vals + (sample[finite_mask] - prev_vals) / c_new
        mean_raw[finite_mask] = updated_vals
        obs_count[finite_mask] = c_new.astype(np.int32)

    state["shape"] = target_shape
    state["mean_raw"] = mean_raw
    state["obs_count"] = obs_count
    state["count"] = int(state.get("count", 0)) + 1
    compare_mask = np.isfinite(prev_mean) & np.isfinite(mean_raw)
    if compare_mask.any():
        diff = mean_raw[compare_mask] - prev_mean[compare_mask]
        delta_l2 = float(np.sqrt(np.sum(diff * diff)))
    else:
        delta_l2 = float("inf")
    state["final_delta_l2"] = delta_l2
    return delta_l2


def _save_map_nodes_csv(map_2d, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_idx", "filter_idx", "value"])
        writer.writeheader()
        if map_2d.size == 0:
            return
        for layer_idx in range(int(map_2d.shape[0])):
            for filter_idx in range(int(map_2d.shape[1])):
                val = float(map_2d[layer_idx, filter_idx]) if np.isfinite(map_2d[layer_idx, filter_idx]) else float("nan")
                writer.writerow({"layer_idx": layer_idx, "filter_idx": filter_idx, "value": val})


def _save_map_nodes_csv_multi(map_by_target, out_path, target_values, layer_indices=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [str(t) for t in target_values]
    fieldnames = ["layer_idx", "filter_idx"] + [f"{t}_grad" for t in targets]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if not targets:
            return
        shapes = []
        for t in targets:
            m = map_by_target.get(t)
            if isinstance(m, np.ndarray):
                shapes.append(m.shape)
        if not shapes:
            return
        h = max(int(s[0]) for s in shapes)
        w = max(int(s[1]) for s in shapes)
        padded = {}
        for t in targets:
            m = map_by_target.get(t)
            if isinstance(m, np.ndarray):
                padded[t] = _pad_map_2d(m, (h, w))
            else:
                padded[t] = np.full((h, w), np.nan, dtype=np.float32)
        li_map = None
        if layer_indices is not None:
            li_map = [int(v) for v in list(layer_indices)]
        for li in range(h):
            layer_idx_out = li_map[li] if (li_map is not None and li < len(li_map)) else li
            for fi in range(w):
                row = {"layer_idx": int(layer_idx_out), "filter_idx": fi}
                for t in targets:
                    v = padded[t][li, fi]
                    row[f"{t}_grad"] = float(v) if np.isfinite(v) else float("nan")
                writer.writerow(row)


def _load_map_nodes_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Map CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"layer_idx", "filter_idx", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Map CSV must contain columns {sorted(required)}: {csv_path}")
    if df.empty:
        return np.zeros((0, 0), dtype=np.float32)
    li = df["layer_idx"].astype(int).to_numpy()
    fi = df["filter_idx"].astype(int).to_numpy()
    vv = df["value"].astype(float).to_numpy()
    h = int(li.max()) + 1
    w = int(fi.max()) + 1
    out = np.full((h, w), np.nan, dtype=np.float32)
    out[li, fi] = vv.astype(np.float32, copy=False)
    return out


def _load_map_nodes_csv_multi(csv_path, target_values=None):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Map CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"layer_idx", "filter_idx"}
    if not required.issubset(df.columns):
        raise ValueError(f"Map CSV must contain columns {sorted(required)}: {csv_path}")
    grad_cols = [c for c in df.columns if c.endswith("_grad")]
    if target_values is not None and len(target_values) > 0:
        expected = [f"{str(t)}_grad" for t in target_values]
        grad_cols = [c for c in expected if c in df.columns]
    if not grad_cols:
        if "value" in df.columns:
            grad_cols = ["value"]
        else:
            return {}
    if df.empty:
        out = {}
        for c in grad_cols:
            key = "value" if c == "value" else c[: -len("_grad")]
            out[key] = np.zeros((0, 0), dtype=np.float32)
        return out
    li = df["layer_idx"].astype(int).to_numpy()
    fi = df["filter_idx"].astype(int).to_numpy()
    h = int(li.max()) + 1
    w = int(fi.max()) + 1
    out = {}
    for c in grad_cols:
        vv = df[c].astype(float).to_numpy()
        m = np.full((h, w), np.nan, dtype=np.float32)
        m[li, fi] = vv.astype(np.float32, copy=False)
        key = "value" if c == "value" else c[: -len("_grad")]
        out[key] = m
    return out


def _resolve_ref_map_path(root_path, map_name):
    root = Path(root_path)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root / "ref_maps" / f"{map_name}.csv"


def _resolve_ref_per_image_noise_dir(root_path):
    root = Path(root_path)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root / "ref_maps" / "per_image" / "noise"


def _pick_target_grad_column(df, target_value):
    tv_col = f"{str(target_value)}_grad"
    if tv_col in df.columns:
        return tv_col
    grad_cols = [c for c in df.columns if c.endswith("_grad")]
    return grad_cols[0] if grad_cols else None


def _build_noise_subspace_models(
    ref_root,
    target_values,
    *,
    centering="centered",
    rank_mode="energy",
    energy_threshold=0.95,
    fixed_k=10,
    max_samples=1000,
):
    noise_root = _resolve_ref_per_image_noise_dir(ref_root)
    if not noise_root.is_dir():
        raise FileNotFoundError(f"Noise per-image directory not found: {noise_root}")

    if rank_mode == "energy":
        allowed = {0.90, 0.95, 0.97, 0.99}
        thr = round(float(energy_threshold), 2)
        if thr not in allowed:
            raise ValueError("subspace.rank.energy_threshold must be one of {0.90,0.95,0.97,0.99}.")
        energy_threshold = thr
    elif rank_mode != "fixed_k":
        raise ValueError("subspace.rank.mode must be one of {'energy','fixed_k'}.")

    out_models = {}
    stats_rows = []
    for tv in target_values:
        target_dir = noise_root / str(tv)
        files = sorted(target_dir.glob("raw_*.csv"))
        if not files:
            raise FileNotFoundError(f"No noise per-image raw csv files found: {target_dir / 'raw_*.csv'}")
        files = files[: max(1, int(max_samples))]

        by_layer_samples = {}
        by_layer_dim = {}
        for fp in files:
            df = pd.read_csv(fp)
            if not {"layer_idx", "filter_idx"}.issubset(df.columns):
                raise ValueError(f"Per-image noise csv missing required columns in: {fp}")
            grad_col = _pick_target_grad_column(df, tv)
            if grad_col is None:
                raise ValueError(f"No '*_grad' column found in: {fp}")
            li = df["layer_idx"].astype(int).to_numpy()
            fi = df["filter_idx"].astype(int).to_numpy()
            vv = df[grad_col].astype(float).to_numpy()
            for layer_idx in np.unique(li):
                mask = li == layer_idx
                f_layer = fi[mask].astype(int, copy=False)
                v_layer = vv[mask].astype(np.float32, copy=False)
                d_here = (int(np.max(f_layer)) + 1) if f_layer.size else 0
                if layer_idx not in by_layer_samples:
                    by_layer_samples[layer_idx] = []
                    by_layer_dim[layer_idx] = d_here
                else:
                    by_layer_dim[layer_idx] = max(int(by_layer_dim[layer_idx]), int(d_here))
                by_layer_samples[layer_idx].append((f_layer, v_layer))

        tv_models = {}
        for layer_idx, sparse_rows in by_layer_samples.items():
            dim = int(by_layer_dim.get(layer_idx, 0))
            n_samples = int(len(sparse_rows))
            if dim <= 0 or n_samples <= 0:
                continue
            G = np.zeros((n_samples, dim), dtype=np.float64)
            for i, (f_idx, vals) in enumerate(sparse_rows):
                if f_idx.size == 0:
                    continue
                m = np.isfinite(vals)
                if m.any():
                    G[i, f_idx[m]] = vals[m].astype(np.float64, copy=False)

            mean_vec = np.zeros((dim,), dtype=np.float64)
            X = G
            if centering == "centered":
                mean_vec = np.mean(G, axis=0)
                X = G - mean_vec[None, :]
            elif centering != "uncentered":
                raise ValueError("subspace.centering must be one of {'centered','uncentered'}.")

            try:
                _u, svals, vt = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            if svals.size == 0 or vt.size == 0:
                continue
            energy = (svals * svals).astype(np.float64, copy=False)
            total_e = float(np.sum(energy))
            if total_e <= 0.0:
                continue
            if rank_mode == "fixed_k":
                k = min(int(fixed_k), int(vt.shape[0]))
            else:
                cum = np.cumsum(energy) / total_e
                k = int(np.searchsorted(cum, float(energy_threshold), side="left") + 1)
                k = min(max(1, k), int(vt.shape[0]))
            basis = vt[:k, :].T.astype(np.float32, copy=False)  # [D, k]
            energy_kept = float(np.sum(energy[:k]) / total_e)
            tv_models[int(layer_idx)] = {
                "basis": basis,
                "mean": mean_vec.astype(np.float32, copy=False),
                "dim": int(dim),
                "rank": int(k),
                "n_samples": int(n_samples),
                "energy_kept": energy_kept,
            }
            stats_rows.append(
                {
                    "target": str(tv),
                    "layer_idx": int(layer_idx),
                    "n_samples": int(n_samples),
                    "dim": int(dim),
                    "rank": int(k),
                    "energy_kept": float(energy_kept),
                }
            )
        out_models[str(tv)] = tv_models

    return out_models, stats_rows


def _apply_subspace_mode_to_map(map_2d, layer_models, mode):
    m = map_2d if map_2d is not None else np.zeros((0, 0), dtype=np.float32)
    if m.ndim != 2 or m.size == 0:
        return m
    if mode not in {"proj", "orth"}:
        raise ValueError("subspace.mode must be one of {'proj','orth'}.")
    rows = []
    max_w = 0
    n_layers = int(m.shape[0])
    for li in range(n_layers):
        row = m[li].astype(np.float32, copy=True)
        model = layer_models.get(int(li))
        if model is None:
            out_row = row
        else:
            D = int(model["dim"])
            B = model["basis"]  # [D,k]
            mu = model["mean"]  # [D]
            x = np.zeros((D,), dtype=np.float32)
            w = min(D, int(row.shape[0]))
            if w > 0:
                vals = row[:w]
                vals = np.where(np.isfinite(vals), vals, 0.0).astype(np.float32, copy=False)
                x[:w] = vals
            xc = x - mu
            proj = (B @ (B.T @ xc)).astype(np.float32, copy=False)
            orth = (xc - proj).astype(np.float32, copy=False)
            if mode == "proj":
                out_row = proj
            else:
                out_row = orth
        rows.append(out_row)
        max_w = max(max_w, int(out_row.shape[0]))
    out = np.full((n_layers, max_w), np.nan, dtype=np.float32)
    for li, row in enumerate(rows):
        if row.size > 0:
            out[li, : row.shape[0]] = row
    return out


def _load_disc_source_maps(fn_non_fn_root, ref_root=None):
    paths = {
        "fn_raw": _resolve_ref_map_path(fn_non_fn_root, "fn_raw_map"),
        "non_fn_raw": _resolve_ref_map_path(fn_non_fn_root, "non_fn_raw_map"),
        "fn_norm": _resolve_ref_map_path(fn_non_fn_root, "fn_norm_map"),
        "non_fn_norm": _resolve_ref_map_path(fn_non_fn_root, "non_fn_norm_map"),
    }
    if ref_root:
        paths["noise_raw"] = _resolve_ref_map_path(ref_root, "noise_raw_map")
        paths["noise_norm"] = _resolve_ref_map_path(ref_root, "noise_norm_map")
    maps = {k: _load_map_nodes_csv(v) for k, v in paths.items()}
    return maps, {k: str(v) for k, v in paths.items()}


def _load_disc_source_maps_multi(fn_non_fn_root, ref_root=None, target_values=None):
    paths = {
        "fn_raw": _resolve_ref_map_path(fn_non_fn_root, "fn_raw_map"),
        "non_fn_raw": _resolve_ref_map_path(fn_non_fn_root, "non_fn_raw_map"),
        "fn_norm": _resolve_ref_map_path(fn_non_fn_root, "fn_norm_map"),
        "non_fn_norm": _resolve_ref_map_path(fn_non_fn_root, "non_fn_norm_map"),
    }
    if ref_root:
        paths["noise_raw"] = _resolve_ref_map_path(ref_root, "noise_raw_map")
        paths["noise_norm"] = _resolve_ref_map_path(ref_root, "noise_norm_map")
    maps = {k: _load_map_nodes_csv_multi(v, target_values=target_values) for k, v in paths.items()}
    return maps, {k: str(v) for k, v in paths.items()}


def _apply_ref_mode_to_map(map_2d, ref_map_2d, ref_mode, eps=1.0e-8):
    m = map_2d if map_2d is not None else np.zeros((0, 0), dtype=np.float32)
    if ref_mode == "none":
        return m
    r = ref_map_2d if ref_map_2d is not None else np.zeros((0, 0), dtype=np.float32)
    target_shape = _merge_map_shape(m.shape, r.shape)
    out = _pad_map_2d(m, target_shape).astype(np.float32, copy=True)
    ref = _pad_map_2d(r, target_shape).astype(np.float32, copy=False)
    if ref_mode == "subtract":
        return out - ref
    if ref_mode == "product":
        return out * ref
    if ref_mode == "proj_removal":
        n_layers = int(out.shape[0]) if out.ndim == 2 else 0
        for i in range(n_layers):
            g_row = out[i]
            r_row = ref[i]
            common = np.isfinite(g_row) & np.isfinite(r_row)
            if not common.any():
                continue
            g = g_row[common].astype(np.float64, copy=False)
            rr = r_row[common].astype(np.float64, copy=False)
            denom = float(np.dot(rr, rr)) + float(eps)
            alpha = float(np.dot(g, rr)) / denom
            g_row[common] = (g - alpha * rr).astype(np.float32, copy=False)
            out[i] = g_row
        return out
    raise ValueError("gradient.ref_corrected must be one of {'none','subtract','product','proj_removal'}.")


def _compute_disc_layer_scores(
    layer_names,
    fn_map,
    non_fn_map,
    ref_map=None,
    *,
    ref_mode="none",
    separation_score="effect_size",
):
    fn_use = fn_map if fn_map is not None else np.zeros((0, 0), dtype=np.float32)
    non_fn_use = non_fn_map if non_fn_map is not None else np.zeros((0, 0), dtype=np.float32)
    if ref_mode in {"subtract", "product", "proj_removal"}:
        ref_use = ref_map if ref_map is not None else np.zeros((0, 0), dtype=np.float32)
        fn_use = _apply_ref_mode_to_map(fn_use, ref_use, ref_mode)
        non_fn_use = _apply_ref_mode_to_map(non_fn_use, ref_use, ref_mode)
    elif ref_mode != "none":
        raise ValueError("gradient.ref_corrected must be one of {'none','subtract','product','proj_removal'}.")

    eps = 1.0e-8
    rows = []
    n_layers = int(min(len(layer_names), fn_use.shape[0] if fn_use.ndim == 2 else 0, non_fn_use.shape[0] if non_fn_use.ndim == 2 else 0))
    for layer_idx in range(n_layers):
        layer_name = str(layer_names[layer_idx])
        fn_row = fn_use[layer_idx]
        non_fn_row = non_fn_use[layer_idx]
        fn_mask = np.isfinite(fn_row)
        non_fn_mask = np.isfinite(non_fn_row)
        common = fn_mask & non_fn_mask
        if not common.any():
            score = float("-inf")
        else:
            fn_vals = fn_row[common].astype(np.float64, copy=False)
            non_fn_vals = non_fn_row[common].astype(np.float64, copy=False)
            diff = fn_vals - non_fn_vals
            mu_dist_l2 = float(np.sqrt(np.sum(diff * diff)))
            var_fn = float(np.var(fn_vals)) if fn_vals.size > 0 else 0.0
            var_non = float(np.var(non_fn_vals)) if non_fn_vals.size > 0 else 0.0
            std_fn = float(np.sqrt(var_fn))
            std_non = float(np.sqrt(var_non))
            if separation_score == "fisher_ratio":
                score = (mu_dist_l2 * mu_dist_l2) / (var_fn + var_non + eps)
            else:
                score = mu_dist_l2 / (std_fn + std_non + eps)
        rows.append({"layer_idx": layer_idx, "layer_name": layer_name, "score": float(score)})
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _load_layer_grad_gt_lookup(gt_csv_path):
    df = pd.read_csv(gt_csv_path)
    required_cols = {"image_id", "image_path", "fn"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"layer_grad.gt CSV must contain columns {sorted(required_cols)}")
    by_id = {}
    by_base = {}
    for _, row in df.iterrows():
        image_id = row.get("image_id")
        image_path = str(row.get("image_path", "")).strip()
        fn = int(row.get("fn", 0))
        if image_id is not None and not pd.isna(image_id):
            by_id[int(image_id)] = fn
        if image_path:
            by_base[Path(image_path).name] = fn
    return by_id, by_base


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = [str(v) for v in parsed["layer_target_values"]]
    target_layers = parsed["layer_target_layers"]
    if isinstance(target_layers, (list, tuple)):
        _disc_tokens = [str(v).strip().lower() for v in target_layers]
    else:
        _disc_tokens = [str(target_layers).strip().lower()]
    disc_enabled = any(tok == "disc_layers" for tok in _disc_tokens)
    ref_type = str(parsed.get("layer_ref_type", "prototype")).strip().lower()
    ref_mode = str(parsed.get("layer_ref_prototype_mode", parsed.get("layer_ref_mode", "none"))).strip().lower()
    ref_subspace_mode = str(parsed.get("layer_ref_subspace_mode", "none")).strip().lower()
    ref_subspace_centering = str(parsed.get("layer_ref_subspace_centering", "centered")).strip().lower()
    ref_subspace_rank_mode = str(parsed.get("layer_ref_subspace_rank_mode", "energy")).strip().lower()
    ref_subspace_energy_threshold = float(parsed.get("layer_ref_subspace_energy_threshold", 0.95))
    ref_subspace_k = max(1, int(parsed.get("layer_ref_subspace_k", 10)))
    ref_subspace_max_samples = max(1, int(parsed.get("layer_ref_subspace_max_samples", 1000)))
    disc_separation_score = str(parsed.get("layer_disc_separation_score", "effect_size")).strip().lower()
    disc_topk = max(1, int(parsed.get("layer_disc_topk", 3)))
    disc_fn_non_fn_map_root = str(parsed.get("layer_disc_fn_non_fn_map_root", "")).strip()
    layer_ref_map_root = str(parsed.get("layer_ref_map_root", "")).strip()
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_vector_reduction = parsed["layer_vector_reduction"]
    layer_pseudo_gt = parsed.get("layer_pseudo_gt", "cand")
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))
    save_image_enabled = bool(parsed.get("save_image_enabled", False))
    per_image_enabled = bool(parsed.get("save_image_layer_grad_per_image_enabled", False))
    per_image_step = max(1, int(parsed.get("save_image_layer_grad_per_image_step", 1)))
    per_image_max_num = max(0, int(parsed.get("save_image_layer_grad_per_image_max_num", 0)))
    image_reference_enabled = bool(parsed.get("save_image_layer_grad_reference_enabled", False))
    image_reference_groups = [g for g in parsed.get("save_image_layer_grad_reference_groups", ["fn", "non_fn"]) if g in {"fn", "non_fn", "noise"}]
    if not image_reference_groups:
        image_reference_groups = ["fn", "non_fn"]
    csv_reference_enabled = bool(parsed.get("save_image_layer_grad_csv_reference_enabled", False))
    csv_reference_groups = [g for g in parsed.get("save_image_layer_grad_csv_reference_groups", ["fn", "non_fn"]) if g in {"fn", "non_fn", "noise"}]
    if not csv_reference_groups:
        csv_reference_groups = ["fn", "non_fn"]
    reference_enabled = bool(image_reference_enabled or csv_reference_enabled)
    reference_groups = image_reference_groups if image_reference_enabled else csv_reference_groups
    used_raw = dataset_cfg.get("used_dataset", [])
    if isinstance(used_raw, str):
        used_list = [used_raw.strip().lower()]
    elif isinstance(used_raw, (list, tuple)):
        used_list = [str(v).strip().lower() for v in used_raw if str(v).strip()]
    else:
        used_list = []
    null_image_mode = "null_image" in used_list
    all_groups = ["noise"] if null_image_mode else ["fn", "non_fn"]
    if null_image_mode:
        reference_groups = ["noise"]
    viz_enabled = bool(unit == "image" and (per_image_enabled or reference_enabled))
    viz_normalize = "layer_minmax"
    viz_target_values = [str(v) for v in list(parsed.get("save_image_layer_grad_target_values", target_values))]
    viz_target_layers = list(parsed.get("save_image_layer_grad_target_layers", target_layers))
    viz_pseudo_gt = str(parsed.get("save_image_layer_grad_pseudo_gt", layer_pseudo_gt)).strip().lower()
    viz_num_by_group = {g: math.inf for g in all_groups}
    viz_gt_csv = str(parsed.get("save_image_layer_grad_gt_csv", "")).strip()
    if image_reference_enabled:
        conv_delta_l2_tol = float(parsed.get("save_image_layer_grad_convergence_delta_l2_tol", 1e-4))
        conv_patience = int(parsed.get("save_image_layer_grad_convergence_patience", 20))
        conv_min_samples = int(parsed.get("save_image_layer_grad_convergence_min_samples", 200))
        conv_max_samples = int(parsed.get("save_image_layer_grad_convergence_max_samples", 20000))
    else:
        conv_delta_l2_tol = float(parsed.get("save_image_layer_grad_csv_convergence_delta_l2_tol", 1e-4))
        conv_patience = int(parsed.get("save_image_layer_grad_csv_convergence_patience", 20))
        conv_min_samples = int(parsed.get("save_image_layer_grad_csv_convergence_min_samples", 200))
        conv_max_samples = int(parsed.get("save_image_layer_grad_csv_convergence_max_samples", 20000))
    convergence_mode = bool(reference_enabled)
    if null_image_mode:
        viz_gt_csv = ""
    if reference_enabled and ("fn" in reference_groups) and not viz_gt_csv:
        raise ValueError("output.layer_grad.reference.gt is required when reference.group includes 'fn'.")
    viz_save_final_raw_map = bool(parsed.get("save_image_layer_grad_save_final_raw_map", True))
    viz_save_final_norm_map = bool(parsed.get("save_image_layer_grad_save_final_norm_map", True))
    viz_save_profile = bool(parsed.get("save_image_layer_grad_save_profile", True))
    viz_save_progress_raw_map = bool(parsed.get("save_image_layer_grad_save_progress_raw_map", False))
    viz_save_progress_norm_map = bool(parsed.get("save_image_layer_grad_save_progress_norm_map", False))
    viz_progress_step = max(1, int(parsed.get("save_image_layer_grad_progress_step", 10)))
    viz_save_reference_per_image_raw_map = bool(parsed.get("save_image_layer_grad_save_reference_per_image_raw_map", False))
    viz_save_reference_per_image_norm_map = bool(parsed.get("save_image_layer_grad_save_reference_per_image_norm_map", False))
    viz_reference_per_image_step = max(1, int(parsed.get("save_image_layer_grad_reference_per_image_step", 10)))
    layer_grad_ref_csv_enabled = bool(parsed.get("save_image_layer_grad_csv_reference_enabled", False))
    layer_grad_ref_save_running_log = bool(parsed.get("save_image_layer_grad_csv_save_running_log", True))
    layer_grad_ref_save_per_image_raw_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_per_image_raw_map_csv", False))
    layer_grad_ref_save_per_image_norm_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_per_image_norm_map_csv", False))
    layer_grad_ref_per_image_step = max(1, int(parsed.get("save_image_layer_grad_csv_per_image_step", 10)))
    layer_grad_ref_save_final_raw_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_final_raw_map_csv", True))
    layer_grad_ref_save_final_norm_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_final_norm_map_csv", True))
    layer_grad_ref_save_progress_raw_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_progress_raw_map_csv", False))
    layer_grad_ref_save_progress_norm_map_csv = bool(parsed.get("save_image_layer_grad_csv_save_progress_norm_map_csv", False))
    layer_grad_ref_progress_step = max(1, int(parsed.get("save_image_layer_grad_csv_progress_step", 10)))
    reference_per_image_image_enabled = bool(
        viz_save_reference_per_image_raw_map or viz_save_reference_per_image_norm_map
    )
    reference_per_image_csv_enabled = bool(
        layer_grad_ref_save_per_image_raw_map_csv or layer_grad_ref_save_per_image_norm_map_csv
    )
    reference_per_image_any_enabled = bool(
        reference_per_image_image_enabled or reference_per_image_csv_enabled
    )

    if not save_csv and not viz_enabled:
        return

    output_csv = run_dir / "layer_grad.csv"

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    all_conv_layers = expand_layer_names(detector.model, ["all_conv"])
    all_conv_name_to_idx = {name: idx for idx, name in enumerate(all_conv_layers)}
    if disc_enabled:
        target_layers = expand_layer_names(detector.model, ["all_conv"])
    else:
        target_layers = expand_layer_names(detector.model, target_layers)
    if not viz_target_values:
        viz_target_values = list(target_values)
    if not viz_target_layers:
        viz_target_layers = list(target_layers)
    viz_target_layers = expand_layer_names(detector.model, viz_target_layers)
    collect_target_values = list(dict.fromkeys(list(target_values) + list(viz_target_values)))
    active_target_values = list(viz_target_values)
    disc_summary = {
        "enabled": bool(disc_enabled),
        "config": {
            "ref_corrected": {
                "mode": ref_type,
                "prototype_mode": ref_mode,
                "subspace_mode": ref_subspace_mode,
            },
            "separation_score": disc_separation_score,
            "topk": int(disc_topk),
            "fn_non_fn_map": disc_fn_non_fn_map_root,
            "ref_map": layer_ref_map_root,
        },
        "selected_layers": [],
    }
    disc_rows = []
    noise_map_for_target_layers_by_target = {}
    subspace_models_by_target = {}
    subspace_stats_rows = []

    print(
        "[INFO] layer_grad gradient options "
        f"(ref_corrected.mode={ref_type}, prototype.mode={ref_mode}, subspace.mode={ref_subspace_mode})"
    )
    if ref_type == "none":
        ref_mode = "none"
    if ref_type == "subspace":
        print(
            "[INFO] layer_grad subspace options "
            f"(centering={ref_subspace_centering}, rank_mode={ref_subspace_rank_mode}, "
            f"energy_threshold={ref_subspace_energy_threshold:.2f}, k={ref_subspace_k}, max_samples={ref_subspace_max_samples})"
        )

    if disc_enabled:
        if unit != "image":
            raise ValueError("gradient.layer='disc_layers' requires output.layer_grad.unit='image'.")
        if ref_type == "subspace":
            raise NotImplementedError("gradient.layer='disc_layers' with ref_corrected.mode='subspace' is not implemented yet.")
        if ref_mode not in {"none", "subtract", "product", "proj_removal"}:
            raise ValueError("gradient.ref_corrected must be one of {'none','subtract','product','proj_removal'}.")
        if not disc_fn_non_fn_map_root:
            raise ValueError("gradient.layer='disc_layers' requires gradient.disc_layers.fn_non_fn_map.")
        if ref_mode != "none" and not layer_ref_map_root:
            raise ValueError("gradient.ref_corrected!='none' requires gradient.ref_map.")
        if disc_separation_score not in {"effect_size", "fisher_ratio"}:
            raise ValueError("gradient.disc_layers.separation_score must be one of {'effect_size','fisher_ratio'}.")

        disc_maps, disc_map_paths = _load_disc_source_maps_multi(
            disc_fn_non_fn_map_root,
            layer_ref_map_root if ref_mode != "none" else None,
            target_values=active_target_values,
        )
        fn_map_by_target = disc_maps["fn_raw"]
        non_fn_map_by_target = disc_maps["non_fn_raw"]
        noise_map_by_target = disc_maps.get("noise_raw", {})

        score_sum_by_layer = {int(i): 0.0 for i in range(len(target_layers))}
        score_by_target = {}
        for tv in active_target_values:
            fn_map_t = fn_map_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
            non_fn_map_t = non_fn_map_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
            noise_map_t = noise_map_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
            rows_t = _compute_disc_layer_scores(
                layer_names=target_layers,
                fn_map=fn_map_t,
                non_fn_map=non_fn_map_t,
                ref_map=noise_map_t,
                ref_mode=ref_mode,
                separation_score=disc_separation_score,
            )
            score_by_target[tv] = {int(r["layer_idx"]): float(r["score"]) for r in rows_t}
            for r in rows_t:
                li = int(r["layer_idx"])
                score_sum_by_layer[li] = float(score_sum_by_layer.get(li, 0.0)) + float(r["score"])
        disc_rows = []
        for li, layer_name in enumerate(target_layers):
            row = {"layer_idx": int(li), "layer_name": str(layer_name), "score": float(score_sum_by_layer.get(li, float("-inf")))}
            for tv in active_target_values:
                row[f"score_{tv}"] = float(score_by_target.get(tv, {}).get(li, float("-inf")))
            disc_rows.append(row)
        disc_rows = sorted(disc_rows, key=lambda x: x["score"], reverse=True)
        for rank, row in enumerate(disc_rows, start=1):
            row["rank"] = int(rank)
        selected_layers = []
        for row in disc_rows:
            if np.isfinite(float(row["score"])):
                selected_layers.append(str(row["layer_name"]))
            if len(selected_layers) >= int(disc_topk):
                break
        if not selected_layers:
            raise ValueError("disc_layers mining failed: no finite separation score was computed.")
        selected_set = set(selected_layers)
        for row in disc_rows:
            row["selected"] = int(row["layer_name"] in selected_set)

        target_layers = list(selected_layers)
        viz_target_layers = list(selected_layers)
        print(
            "[INFO] discriminative layers "
            f"(ref_corrected.mode={ref_type}, prototype.mode={ref_mode}, score={disc_separation_score}, topk={int(disc_topk)}): "
            + ", ".join(selected_layers)
        )
        selected_indices = [int(row["layer_idx"]) for row in disc_rows if int(row.get("selected", 0)) == 1]
        if ref_mode != "none":
            for tv in active_target_values:
                nm = noise_map_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
                if selected_indices and nm.ndim == 2 and max(selected_indices) < int(nm.shape[0]):
                    noise_map_for_target_layers_by_target[tv] = nm[selected_indices, :].astype(np.float32, copy=True)
                else:
                    noise_map_for_target_layers_by_target[tv] = np.zeros((len(selected_indices), 0), dtype=np.float32)
        disc_summary["map_paths"] = disc_map_paths
        disc_summary["selected_layers"] = list(selected_layers)
    elif ref_type == "subspace":
        if unit != "image":
            raise ValueError("gradient.ref_corrected.mode='subspace' requires output.layer_grad.unit='image'.")
        if not layer_ref_map_root:
            raise ValueError("gradient.ref_corrected.mode='subspace' requires gradient.ref_corrected.ref_map.")
        if ref_subspace_mode not in {"proj", "orth"}:
            raise ValueError("gradient.ref_corrected.subspace.mode must be one of {'proj','orth'}.")
        subspace_models_by_target, subspace_stats_rows = _build_noise_subspace_models(
            layer_ref_map_root,
            target_values=target_values,
            centering=ref_subspace_centering,
            rank_mode=ref_subspace_rank_mode,
            energy_threshold=ref_subspace_energy_threshold,
            fixed_k=ref_subspace_k,
            max_samples=ref_subspace_max_samples,
        )
        if not subspace_stats_rows:
            raise ValueError("No valid subspace model could be built from noise per-image CSVs.")
        for r in subspace_stats_rows[:8]:
            print(
                "[INFO] subspace "
                f"target={r['target']} layer_idx={r['layer_idx']} n={r['n_samples']} d={r['dim']} "
                f"rank={r['rank']} energy={r['energy_kept']:.4f}"
            )
    elif ref_mode != "none":
        if ref_mode not in {"subtract", "product", "proj_removal"}:
            raise ValueError("gradient.ref_corrected must be one of {'none','subtract','product','proj_removal'}.")
        if not layer_ref_map_root:
            raise ValueError("gradient.ref_corrected!='none' requires gradient.ref_map.")
        target_indices = [int(all_conv_name_to_idx[name]) for name in target_layers if name in all_conv_name_to_idx]
        noise_map_name = "noise_raw_map"
        noise_map_by_target = _load_map_nodes_csv_multi(
            _resolve_ref_map_path(layer_ref_map_root, noise_map_name),
            target_values=active_target_values,
        )
        for tv in active_target_values:
            nm = noise_map_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
            if target_indices and nm.ndim == 2 and max(target_indices) < int(nm.shape[0]):
                noise_map_for_target_layers_by_target[tv] = nm[target_indices, :].astype(np.float32, copy=True)
            else:
                noise_map_for_target_layers_by_target[tv] = np.zeros((len(target_indices), 0), dtype=np.float32)

    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(["pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"])
    for target_value in target_values:
        for layer_name in target_layers:
            fieldnames.append(f"{target_value}_{layer_name}")

    layer_param_shapes = {}
    viz_target_layer_indices = [
        int(all_conv_name_to_idx[layer_name])
        for layer_name in viz_target_layers
        if layer_name in all_conv_name_to_idx
    ]
    need_transform_maps = bool(save_csv and unit == "image" and ((ref_mode != "none") or (ref_type == "subspace")))
    if unit == "image" and (viz_enabled or need_transform_maps):
        shape_layers = list(dict.fromkeys(list(viz_target_layers) + list(target_layers)))
        for layer_name in shape_layers:
            try:
                layer_param_shapes[layer_name] = tuple(resolve_layer_parameter(detector.model, layer_name).shape)
            except Exception:
                layer_param_shapes[layer_name] = None
    catid_to_name = load_gt_category_maps(config, split) if viz_enabled else {}
    iou_match_threshold = parsed["gt_iou_match_threshold"] if viz_enabled else 0.45
    per_image_seen = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    per_image_saved = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_progress_image_saved = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_per_image_seen = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_per_image_saved = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_per_image_csv_seen = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_per_image_csv_saved = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    ref_progress_csv_saved = {g: {tv: 0 for tv in active_target_values} for g in all_groups}
    tb_writer = None
    tb_log_dir = None
    gt_match_stats = {"id_match": 0, "path_fallback": 0, "unmatched": 0}
    gt_by_id, gt_by_base = {}, {}
    if viz_gt_csv:
        gt_path = Path(viz_gt_csv)
        if not gt_path.is_absolute():
            gt_path = (Path.cwd() / gt_path).resolve()
        gt_by_id, gt_by_base = _load_layer_grad_gt_lookup(gt_path)

    def _make_target_state():
        return {
            "count": 0,
            "mean_raw": None,
            "obs_count": None,
            "shape": None,
            "stable_steps": 0,
            "converged": False,
            "done": False,
            "final_delta_l2": float("inf"),
            "stop_reason": "",
        }

    def _make_group_state():
        return {
            "targets": {tv: _make_target_state() for tv in active_target_values},
            "done": False,
            "stop_reason": "",
        }

    group_states = {g: _make_group_state() for g in all_groups}
    active_reference_groups = [g for g in all_groups if g in reference_groups]

    def _is_target_done(group_key, target_key):
        gst = group_states[group_key]
        st = gst["targets"][target_key]
        if st["done"]:
            return True
        target_num = viz_num_by_group[group_key]
        if not np.isinf(target_num):
            if st["count"] >= int(target_num):
                st["done"] = True
                st["stop_reason"] = "target_reached"
                return True
            return False
        if st["converged"]:
            st["done"] = True
            st["stop_reason"] = "converged"
            return True
        if st["count"] >= conv_max_samples:
            st["done"] = True
            st["stop_reason"] = "max_samples_reached"
            return True
        return False

    def _is_group_done(group_key):
        if reference_enabled:
            gst = group_states[group_key]
            if gst["done"]:
                return True
            all_done = True
            for tv in active_target_values:
                if not _is_target_done(group_key, tv):
                    all_done = False
            if all_done:
                gst["done"] = True
                if all(gst["targets"][tv]["stop_reason"] == "converged" for tv in active_target_values):
                    gst["stop_reason"] = "converged"
                elif all(gst["targets"][tv]["stop_reason"] == "target_reached" for tv in active_target_values):
                    gst["stop_reason"] = "target_reached"
                else:
                    gst["stop_reason"] = "mixed"
                return True
            return False
        if per_image_enabled and per_image_max_num > 0:
            return all(per_image_saved[group_key][tv] >= per_image_max_num for tv in active_target_values)
        return False

    def _all_done():
        if reference_enabled:
            if reference_per_image_any_enabled:
                return False
            return all(_is_group_done(g) for g in active_reference_groups)
        return all(_is_group_done(g) for g in all_groups)

    viz_dir = run_dir / "images"
    if viz_enabled:
        viz_dir.mkdir(parents=True, exist_ok=True)
    if viz_enabled and per_image_enabled:
        for g in all_groups:
            for tv in active_target_values:
                (viz_dir / "per_image" / g / tv).mkdir(parents=True, exist_ok=True)
    if viz_enabled and reference_enabled and (viz_save_progress_raw_map or viz_save_progress_norm_map):
        for g in all_groups:
            for tv in active_target_values:
                (viz_dir / "reference_progress" / g / tv).mkdir(parents=True, exist_ok=True)
    if viz_enabled and reference_enabled and (viz_save_reference_per_image_raw_map or viz_save_reference_per_image_norm_map):
        for g in all_groups:
            for tv in active_target_values:
                (viz_dir / "reference_per_image" / g / tv).mkdir(parents=True, exist_ok=True)
    if viz_enabled and reference_enabled and layer_grad_ref_csv_enabled and layer_grad_ref_save_running_log:
        tb_log_dir = run_dir / "ref_maps" / "tensorboard"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    csv_file_handle = None
    csv_writer = None
    if save_csv:
        csv_file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            if _all_done():
                break
            image_list = _as_image_list(images)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            batch_preds = None
            batch_grad_stats_all = None
            required_layers = []
            if unit != "bbox":
                if csv_writer is not None:
                    required_layers.extend(target_layers)
                if viz_enabled:
                    required_layers.extend(viz_target_layers)
                required_layers = list(dict.fromkeys(required_layers))
                if required_layers:
                    batch_grad_stats_all = collect_batch_image_layer_grads_per_target(
                        detector=detector,
                        input_tensor=infer_batch,
                        target_values=collect_target_values,
                        target_layers=required_layers,
                        map_reduction=layer_map_reduction,
                        vector_reduction=[],
                        pre_nms=pre_nms,
                        pre_nms_ratio=pre_nms_ratio,
                        pseudo_gt=viz_pseudo_gt if viz_enabled else layer_pseudo_gt,
                    )
                else:
                    batch_grad_stats_all = [{} for _ in range(len(image_list))]

            for sample_idx in range(len(image_list)):
                if _all_done():
                    break
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                infer_tensor = infer_batch[sample_idx: sample_idx + 1]
                if unit == "bbox":
                    bbox_rows = collect_bbox_layer_grads_per_target(
                        detector=detector,
                        input_tensor=infer_tensor,
                        target_values=target_values,
                        target_layers=target_layers,
                        map_reduction=layer_map_reduction,
                        vector_reduction=layer_vector_reduction,
                        pseudo_gt=layer_pseudo_gt,
                    )
                    if csv_writer is not None:
                        for bbox_row in bbox_rows:
                            row = {
                                "image_id": image_id,
                                "image_path": image_path,
                                "pred_idx": bbox_row["pred_idx"],
                                "raw_pred_idx": bbox_row["raw_pred_idx"],
                                "xmin": bbox_row["xmin"],
                                "ymin": bbox_row["ymin"],
                                "xmax": bbox_row["xmax"],
                                "ymax": bbox_row["ymax"],
                                "score": bbox_row["score"],
                                "pred_class": bbox_row["pred_class"],
                            }
                            for grad_key, grad_value in bbox_row["grad_stats"].items():
                                row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                            csv_writer.writerow(row)
                    del bbox_rows
                else:
                    group_key = None
                    fn_flag = None
                    st = None
                    if viz_enabled:
                        if null_image_mode:
                            group_key = "noise"
                        else:
                            if viz_gt_csv:
                                if image_id in gt_by_id:
                                    fn_flag = int(gt_by_id[image_id])
                                    gt_match_stats["id_match"] += 1
                                else:
                                    base_name = Path(str(image_path)).name
                                    if base_name in gt_by_base:
                                        fn_flag = int(gt_by_base[base_name])
                                        gt_match_stats["path_fallback"] += 1
                                    else:
                                        gt_match_stats["unmatched"] += 1
                                        if convergence_mode:
                                            continue
                            if fn_flag is None:
                                if batch_preds is None:
                                    detector.zero_grad(set_to_none=True)
                                    with torch.no_grad():
                                        batch_preds, _bz_logits, _bz_obj, _bz_feat = detector(infer_batch)
                                pred_boxes = batch_preds[0][sample_idx]
                                pred_class_names = batch_preds[2][sample_idx]
                                gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                                is_fn = has_fn_for_image(
                                    gt_boxes=gt_boxes,
                                    gt_class_names=gt_class_names,
                                    pred_boxes=pred_boxes,
                                    pred_class_names=pred_class_names,
                                    iou_match_threshold=iou_match_threshold,
                                )
                                fn_flag = int(is_fn)
                            group_key = "fn" if int(fn_flag) == 1 else "non_fn"
                        st = group_states[group_key]
                        if (
                            reference_enabled
                            and (group_key in active_reference_groups)
                            and _is_group_done(group_key)
                            and not reference_per_image_any_enabled
                        ):
                            continue
                    grad_stats_all = batch_grad_stats_all[sample_idx] if batch_grad_stats_all is not None else {}

                    if csv_writer is not None:
                        row = {"image_id": image_id, "image_path": image_path}
                        transformed_maps_by_target = {}
                        if need_transform_maps:
                            for target_value in target_values:
                                map_raw_t = _build_layer_filter_map_from_grad_stats(
                                    grad_stats=grad_stats_all,
                                    target_values=[target_value],
                                    target_layers=target_layers,
                                    layer_param_shapes=layer_param_shapes,
                                )
                                map_use_t = map_raw_t
                                if ref_type == "subspace":
                                    map_use_t = _apply_subspace_mode_to_map(
                                        map_use_t,
                                        subspace_models_by_target.get(str(target_value), {}),
                                        mode=ref_subspace_mode,
                                    )
                                else:
                                    map_use_t = _apply_ref_mode_to_map(
                                        map_use_t,
                                        noise_map_for_target_layers_by_target.get(str(target_value)),
                                        ref_mode,
                                    )
                                transformed_maps_by_target[target_value] = map_use_t
                        for target_value in target_values:
                            for layer_idx, layer_name in enumerate(target_layers):
                                grad_key = f"{target_value}_{layer_name}"
                                if need_transform_maps:
                                    map_t = transformed_maps_by_target.get(target_value, np.zeros((0, 0), dtype=np.float32))
                                    if map_t.ndim == 2 and layer_idx < int(map_t.shape[0]):
                                        vec_np = map_t[layer_idx]
                                        vec_np = vec_np[np.isfinite(vec_np)].astype(np.float32, copy=False)
                                        grad_value = vec_np.tolist()
                                    else:
                                        grad_value = []
                                else:
                                    grad_value = grad_stats_all.get(grad_key, [])
                                if layer_vector_reduction:
                                    vec = torch.tensor(_vector_from_grad_value(grad_value), dtype=torch.float32)
                                    stats = map_grad_tensor_to_numbers(vec)
                                    row[grad_key] = json.dumps(
                                        {k: float(stats[k]) for k in layer_vector_reduction},
                                        separators=(",", ":"),
                                    )
                                else:
                                    if isinstance(grad_value, torch.Tensor):
                                        grad_value = grad_value.detach().float().cpu().tolist()
                                    row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                        csv_writer.writerow(row)

                    if viz_enabled and group_key is not None:
                        grad_map_raw_by_target = _build_layer_filter_map_by_target_from_grad_stats(
                            grad_stats=grad_stats_all,
                            target_values=viz_target_values,
                            target_layers=viz_target_layers,
                            layer_param_shapes=layer_param_shapes,
                        )
                        if reference_enabled and (group_key in active_reference_groups):
                            for tv in active_target_values:
                                st_t = group_states[group_key]["targets"][tv]
                                map_t = grad_map_raw_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
                                if layer_grad_ref_csv_enabled and (layer_grad_ref_save_per_image_raw_map_csv or layer_grad_ref_save_per_image_norm_map_csv):
                                    ref_per_image_csv_seen[group_key][tv] += 1
                                    should_save_per_image_csv = (
                                        (ref_per_image_csv_seen[group_key][tv] - 1) % int(layer_grad_ref_per_image_step)
                                    ) == 0
                                    if should_save_per_image_csv:
                                        per_idx = int(ref_per_image_csv_saved[group_key][tv])
                                        ref_per_img_dir = run_dir / "ref_maps" / "per_image" / group_key / tv
                                        ref_per_img_dir.mkdir(parents=True, exist_ok=True)
                                        map_dict_one = {tv: map_t}
                                        if layer_grad_ref_save_per_image_raw_map_csv:
                                            _save_map_nodes_csv_multi(
                                                map_dict_one,
                                                ref_per_img_dir / f"raw_{per_idx:05d}.csv",
                                                target_values=[tv],
                                                layer_indices=viz_target_layer_indices,
                                            )
                                        if layer_grad_ref_save_per_image_norm_map_csv:
                                            _save_map_nodes_csv_multi(
                                                {tv: _normalize_layer_map(map_t, mode=viz_normalize)},
                                                ref_per_img_dir / f"norm_{per_idx:05d}.csv",
                                                target_values=[tv],
                                                layer_indices=viz_target_layer_indices,
                                            )
                                        ref_per_image_csv_saved[group_key][tv] += 1
                                if viz_save_reference_per_image_raw_map or viz_save_reference_per_image_norm_map:
                                    ref_per_image_seen[group_key][tv] += 1
                                    should_save_reference_per_image = (
                                        (ref_per_image_seen[group_key][tv] - 1) % int(viz_reference_per_image_step)
                                    ) == 0
                                    if should_save_reference_per_image:
                                        per_idx = int(ref_per_image_saved[group_key][tv])
                                        if viz_save_reference_per_image_raw_map:
                                            out_raw = viz_dir / "reference_per_image" / group_key / tv / f"raw_{per_idx:05d}.png"
                                            _save_heatmap_png(map_t, out_raw)
                                        if viz_save_reference_per_image_norm_map:
                                            out_norm = viz_dir / "reference_per_image" / group_key / tv / f"norm_{per_idx:05d}.png"
                                            _save_heatmap_png(_normalize_layer_map(map_t, mode=viz_normalize), out_norm)
                                        ref_per_image_saved[group_key][tv] += 1
                                if _is_target_done(group_key, tv):
                                    continue
                                delta_l2 = _update_running_mean_map(st_t, map_t)
                                if np.isinf(viz_num_by_group[group_key]):
                                    if (
                                        st_t["count"] >= conv_min_samples
                                        and np.isfinite(delta_l2)
                                        and delta_l2 <= conv_delta_l2_tol
                                    ):
                                        st_t["stable_steps"] += 1
                                    else:
                                        st_t["stable_steps"] = 0
                                    if st_t["stable_steps"] >= conv_patience:
                                        st_t["converged"] = True
                                _is_target_done(group_key, tv)
                                if layer_grad_ref_csv_enabled and layer_grad_ref_save_running_log and tb_writer is not None:
                                    step_val = int(st_t["count"])
                                    tb_writer.add_scalar(f"layer_grad/{group_key}/{tv}/delta_l2", float(delta_l2), step_val)
                                    tb_writer.add_scalar(f"layer_grad/{group_key}/{tv}/converged", int(bool(st_t["converged"])), step_val)
                                if viz_save_progress_raw_map or viz_save_progress_norm_map:
                                    should_save_progress_img = ((int(st_t["count"]) % int(viz_progress_step)) == 0)
                                    if should_save_progress_img and st_t.get("mean_raw") is not None:
                                        progress_idx = int(ref_progress_image_saved[group_key][tv])
                                        if viz_save_progress_raw_map:
                                            out_raw = viz_dir / "reference_progress" / group_key / tv / f"raw_{progress_idx:05d}.png"
                                            _save_heatmap_png(st_t["mean_raw"], out_raw)
                                        if viz_save_progress_norm_map:
                                            out_norm = viz_dir / "reference_progress" / group_key / tv / f"norm_{progress_idx:05d}.png"
                                            _save_heatmap_png(_normalize_layer_map(st_t["mean_raw"], mode=viz_normalize), out_norm)
                                        ref_progress_image_saved[group_key][tv] += 1
                                if layer_grad_ref_csv_enabled and (layer_grad_ref_save_progress_raw_map_csv or layer_grad_ref_save_progress_norm_map_csv):
                                    should_save_progress_csv = ((int(st_t["count"]) % int(layer_grad_ref_progress_step)) == 0)
                                    if should_save_progress_csv and st_t.get("mean_raw") is not None:
                                        progress_idx = int(ref_progress_csv_saved[group_key][tv])
                                        ref_prog_dir = run_dir / "ref_maps" / "progress" / group_key
                                        ref_prog_dir.mkdir(parents=True, exist_ok=True)
                                        map_dict_all = {}
                                        for tt in active_target_values:
                                            m_tt = group_states[group_key]["targets"][tt]["mean_raw"]
                                            map_dict_all[tt] = m_tt if m_tt is not None else np.zeros((0, 0), dtype=np.float32)
                                        if layer_grad_ref_save_progress_raw_map_csv:
                                            _save_map_nodes_csv_multi(
                                                map_dict_all,
                                                ref_prog_dir / f"{tv}_raw_{progress_idx:05d}.csv",
                                                target_values=active_target_values,
                                                layer_indices=viz_target_layer_indices,
                                            )
                                        if layer_grad_ref_save_progress_norm_map_csv:
                                            norm_dict_all = {tt: _normalize_layer_map(map_dict_all[tt], mode=viz_normalize) for tt in active_target_values}
                                            _save_map_nodes_csv_multi(
                                                norm_dict_all,
                                                ref_prog_dir / f"{tv}_norm_{progress_idx:05d}.csv",
                                                target_values=active_target_values,
                                                layer_indices=viz_target_layer_indices,
                                            )
                                        ref_progress_csv_saved[group_key][tv] += 1
                            _is_group_done(group_key)
                        if per_image_enabled:
                            for tv in active_target_values:
                                per_image_seen[group_key][tv] += 1
                                should_save = ((per_image_seen[group_key][tv] - 1) % per_image_step == 0)
                                if should_save and (per_image_max_num <= 0 or per_image_saved[group_key][tv] < per_image_max_num):
                                    m = grad_map_raw_by_target.get(tv, np.zeros((0, 0), dtype=np.float32))
                                    out_path = viz_dir / "per_image" / group_key / tv / f"{image_id}_{per_image_saved[group_key][tv]:05d}.png"
                                    _save_heatmap_png(_normalize_layer_map(m, mode=viz_normalize), out_path)
                                    per_image_saved[group_key][tv] += 1
                    del grad_stats_all
            del infer_batch
            if batch_preds is not None:
                del batch_preds
            if batch_grad_stats_all is not None:
                del batch_grad_stats_all
    finally:
        if csv_file_handle is not None:
            csv_file_handle.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
            tb_writer = None

    if viz_enabled:
        if reference_enabled:
            for g in all_groups:
                gst = group_states[g]
                for tv in active_target_values:
                    st_t = gst["targets"][tv]
                    if st_t["stop_reason"] == "":
                        if st_t["converged"]:
                            st_t["stop_reason"] = "converged"
                        elif st_t["done"]:
                            st_t["stop_reason"] = "target_reached"
                        else:
                            st_t["stop_reason"] = "dataloader_exhausted"
                _is_group_done(g)
            if viz_save_final_raw_map:
                for g in all_groups:
                    for tv in active_target_values:
                        m = group_states[g]["targets"][tv]["mean_raw"]
                        if m is not None:
                            _save_heatmap_png(m, viz_dir / f"{g}_final_raw_{tv}_map.png")
            if viz_save_final_norm_map:
                for g in all_groups:
                    for tv in active_target_values:
                        m = group_states[g]["targets"][tv]["mean_raw"]
                        if m is None:
                            continue
                        _save_heatmap_png(
                            _normalize_layer_map(m, mode=viz_normalize),
                            viz_dir / f"{g}_final_norm_{tv}_map.png",
                        )
            if (not null_image_mode) and viz_save_profile:
                for tv in active_target_values:
                    fn_m = group_states.get("fn", {}).get("targets", {}).get(tv, {}).get("mean_raw")
                    non_m = group_states.get("non_fn", {}).get("targets", {}).get(tv, {}).get("mean_raw")
                    fn_norm = _normalize_layer_map(fn_m, mode=viz_normalize) if fn_m is not None else np.zeros((0, 0), dtype=np.float32)
                    non_norm = _normalize_layer_map(non_m, mode=viz_normalize) if non_m is not None else np.zeros((0, 0), dtype=np.float32)
                    if fn_norm.size > 0 or non_norm.size > 0:
                        _save_layer_profile_plot(
                            fn_mean_map=fn_norm,
                            non_fn_mean_map=non_norm,
                            out_path=viz_dir / f"profile_{tv}_mean_std_log.png",
                            log_scale=False,
                        )
            if layer_grad_ref_csv_enabled:
                ref_dir = run_dir / "ref_maps"
                ref_dir.mkdir(parents=True, exist_ok=True)
                if layer_grad_ref_save_final_raw_map_csv:
                    for g in all_groups:
                        map_dict = {}
                        for tv in active_target_values:
                            m = group_states[g]["targets"][tv]["mean_raw"]
                            map_dict[tv] = m if m is not None else np.zeros((0, 0), dtype=np.float32)
                        _save_map_nodes_csv_multi(
                            map_dict,
                            ref_dir / f"{g}_raw_map.csv",
                            target_values=active_target_values,
                            layer_indices=viz_target_layer_indices,
                        )
                if layer_grad_ref_save_final_norm_map_csv:
                    for g in all_groups:
                        norm_dict = {}
                        for tv in active_target_values:
                            m = group_states[g]["targets"][tv]["mean_raw"]
                            norm_dict[tv] = _normalize_layer_map(m, mode=viz_normalize) if m is not None else np.zeros((0, 0), dtype=np.float32)
                        _save_map_nodes_csv_multi(
                            norm_dict,
                            ref_dir / f"{g}_norm_map.csv",
                            target_values=active_target_values,
                            layer_indices=viz_target_layer_indices,
                        )
        if gt_match_stats.get("unmatched", 0) > 0:
            print(f"[layer_grad] unmatched GT rows: {int(gt_match_stats['unmatched'])}")
        group_total_counts = {}
        group_converged = {}
        group_final_delta_l2 = {}
        group_stable_steps = {}
        group_stop_reason = {}
        group_target_converged = {}
        group_target_final_delta_l2 = {}
        group_target_stable_steps = {}
        group_target_stop_reason = {}
        for g in all_groups:
            gst = group_states[g]
            target_counts = {tv: int(gst["targets"][tv]["count"]) for tv in active_target_values}
            target_deltas = {
                tv: (float(gst["targets"][tv]["final_delta_l2"]) if np.isfinite(gst["targets"][tv]["final_delta_l2"]) else None)
                for tv in active_target_values
            }
            target_stables = {tv: int(gst["targets"][tv]["stable_steps"]) for tv in active_target_values}
            target_conv = {tv: bool(gst["targets"][tv]["converged"]) for tv in active_target_values}
            target_stop = {tv: str(gst["targets"][tv]["stop_reason"]) for tv in active_target_values}
            group_target_converged[g] = target_conv
            group_target_final_delta_l2[g] = target_deltas
            group_target_stable_steps[g] = target_stables
            group_target_stop_reason[g] = target_stop
            group_total_counts[g] = int(max(target_counts.values()) if target_counts else 0)
            group_converged[g] = bool(all(target_conv.values())) if target_conv else False
            finite_vals = [v for v in target_deltas.values() if v is not None]
            group_final_delta_l2[g] = (float(np.mean(np.asarray(finite_vals, dtype=np.float32))) if finite_vals else None)
            group_stable_steps[g] = int(min(target_stables.values()) if target_stables else 0)
            group_stop_reason[g] = str(gst.get("stop_reason", ""))
            if group_stop_reason[g] == "":
                if all(v == "dataloader_exhausted" for v in target_stop.values()):
                    group_stop_reason[g] = "dataloader_exhausted"
        viz_summary = {
            "normalize": viz_normalize,
            "mode": "reference" if reference_enabled else "per_image",
            "num_by_group": {k: ("inf" if np.isinf(viz_num_by_group[k]) else int(viz_num_by_group[k])) for k in all_groups},
            "target_values_for_map": list(active_target_values),
            "convergence": {
                "delta_metric": "l2",
                "delta_l2_tol": conv_delta_l2_tol,
                "patience": conv_patience,
                "min_samples": conv_min_samples,
                "max_samples": conv_max_samples,
            },
            "null_image_mode": bool(null_image_mode),
            "group_total_counts": group_total_counts,
            "group_converged": group_converged,
            "group_final_delta_l2": group_final_delta_l2,
            "group_stable_steps": group_stable_steps,
            "group_stop_reason": group_stop_reason,
            "group_target_converged": group_target_converged,
            "group_target_final_delta_l2": group_target_final_delta_l2,
            "group_target_stable_steps": group_target_stable_steps,
            "group_target_stop_reason": group_target_stop_reason,
            "target_layers_for_map": viz_target_layers,
            "inferenced_image": {
                "enabled": bool(per_image_enabled),
                "step": int(per_image_step),
                "max_num": int(per_image_max_num),
                "saved": {k: {tv: int(per_image_saved[k][tv]) for tv in active_target_values} for k in all_groups},
            },
            "reference_enabled": bool(reference_enabled),
            "reference_groups": active_reference_groups,
            "reference_csv_enabled": bool(layer_grad_ref_csv_enabled),
            "reference_progress_image": {
                "save_raw": bool(viz_save_progress_raw_map),
                "save_norm": bool(viz_save_progress_norm_map),
                "step": int(viz_progress_step),
                "saved": {k: {tv: int(ref_progress_image_saved[k][tv]) for tv in active_target_values} for k in all_groups},
            },
            "reference_per_image": {
                "save_raw": bool(viz_save_reference_per_image_raw_map),
                "save_norm": bool(viz_save_reference_per_image_norm_map),
                "step": int(viz_reference_per_image_step),
                "saved": {k: {tv: int(ref_per_image_saved[k][tv]) for tv in active_target_values} for k in all_groups},
            },
            "reference_progress_csv": {
                "save_raw_csv": bool(layer_grad_ref_save_progress_raw_map_csv),
                "save_norm_csv": bool(layer_grad_ref_save_progress_norm_map_csv),
                "step": int(layer_grad_ref_progress_step),
                "saved": {k: {tv: int(ref_progress_csv_saved[k][tv]) for tv in active_target_values} for k in all_groups},
            },
            "reference_per_image_csv": {
                "save_raw_csv": bool(layer_grad_ref_save_per_image_raw_map_csv),
                "save_norm_csv": bool(layer_grad_ref_save_per_image_norm_map_csv),
                "step": int(layer_grad_ref_per_image_step),
                "saved": {k: {tv: int(ref_per_image_csv_saved[k][tv]) for tv in active_target_values} for k in all_groups},
            },
            "tensorboard_log_dir": str(tb_log_dir) if tb_log_dir is not None else "",
            "save_final_raw_map": bool(viz_save_final_raw_map),
            "save_final_norm_map": bool(viz_save_final_norm_map),
            "save_profile": bool(viz_save_profile),
            "gt_match_stats": gt_match_stats,
            "disc_layers": disc_summary,
            "ref_corrected": {
                "mode": ref_type,
                "prototype_mode": ref_mode,
                "subspace_mode": ref_subspace_mode,
                "subspace": {
                    "config": {
                        "centering": ref_subspace_centering,
                        "rank_mode": ref_subspace_rank_mode,
                        "energy_threshold": ref_subspace_energy_threshold,
                        "k": ref_subspace_k,
                        "max_samples": ref_subspace_max_samples,
                        "ref_map": layer_ref_map_root,
                    },
                    "stats": subspace_stats_rows,
                },
            },
        }
        with open(viz_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(viz_summary, f, ensure_ascii=False, indent=2)
    if disc_enabled:
        ref_dir = run_dir / "ref_maps"
        ref_dir.mkdir(parents=True, exist_ok=True)
        disc_csv_path = ref_dir / "disc_layer_scores.csv"
        extra_fields = [f"score_{tv}" for tv in active_target_values]
        with open(disc_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "layer_idx", "layer_name", "score"] + extra_fields + ["selected"],
            )
            writer.writeheader()
            for row in disc_rows:
                out = {
                    "rank": int(row.get("rank", 0)),
                    "layer_idx": int(row.get("layer_idx", -1)),
                    "layer_name": str(row.get("layer_name", "")),
                    "score": float(row.get("score", float("nan"))),
                    "selected": int(row.get("selected", 0)),
                }
                for tv in active_target_values:
                    out[f"score_{tv}"] = float(row.get(f"score_{tv}", float("nan")))
                writer.writerow(out)
    if ref_type == "subspace":
        ref_dir = run_dir / "ref_maps"
        ref_dir.mkdir(parents=True, exist_ok=True)
        subspace_csv = ref_dir / "subspace_summary.csv"
        with open(subspace_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["target", "layer_idx", "n_samples", "dim", "rank", "energy_kept"],
            )
            writer.writeheader()
            for r in subspace_stats_rows:
                writer.writerow(
                    {
                        "target": str(r.get("target", "")),
                        "layer_idx": int(r.get("layer_idx", -1)),
                        "n_samples": int(r.get("n_samples", 0)),
                        "dim": int(r.get("dim", 0)),
                        "rank": int(r.get("rank", 0)),
                        "energy_kept": float(r.get("energy_kept", float("nan"))),
                    }
                )
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    if viz_enabled:
        print(f"Saved layer-grad maps: {viz_dir}")
        if tb_log_dir is not None:
            print(f"Saved layer-grad TensorBoard logs: {tb_log_dir}")



__all__ = [
    "run_layer_grad_csv",
    "_build_layer_filter_map_from_grad_stats",
    "_build_layer_filter_map_by_target_from_grad_stats",
    "_aggregate_target_maps",
    "_normalize_layer_map",
    "_save_layer_profile_plot",
    "_save_heatmap_png",
    "_pad_map_2d",
    "_pad_count_2d",
    "_merge_map_shape",
    "_update_running_mean_map",
    "_save_map_nodes_csv",
    "_save_map_nodes_csv_multi",
    "_load_map_nodes_csv",
    "_load_map_nodes_csv_multi",
    "_resolve_ref_map_path",
    "_load_disc_source_maps",
    "_load_disc_source_maps_multi",
    "_apply_ref_mode_to_map",
    "_compute_disc_layer_scores",
    "_load_layer_grad_gt_lookup",
]
