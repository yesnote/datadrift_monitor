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


def _save_map_nodes_csv_multi(map_by_target, out_path, target_values):
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
        for li in range(h):
            for fi in range(w):
                row = {"layer_idx": li, "filter_idx": fi}
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
    raise ValueError("gradient.ref_corrected must be one of {'none','subtract','proj_removal'}.")


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
    if ref_mode in {"subtract", "proj_removal"}:
        ref_use = ref_map if ref_map is not None else np.zeros((0, 0), dtype=np.float32)
        fn_use = _apply_ref_mode_to_map(fn_use, ref_use, ref_mode)
        non_fn_use = _apply_ref_mode_to_map(non_fn_use, ref_use, ref_mode)
    elif ref_mode != "none":
        raise ValueError("gradient.ref_corrected must be one of {'none','subtract','proj_removal'}.")

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
    from object_detectors.commands.run_predict_impl import run_layer_grad_csv as _impl

    return _impl(config, run_dir)


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
