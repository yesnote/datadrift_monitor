import torch

from object_detectors.commands.predict import (
    run_energy_csv,
    run_ensemble_csv,
    run_entropy_csv,
    run_feature_csv,
    run_feature_grad_csv,
    run_fn_csv,
    run_full_softmax_csv,
    run_layer_grad_csv,
    run_mc_dropout_csv,
    run_meta_detect_csv,
    run_score_csv,
    run_tp_csv,
)
from object_detectors.commands.utils.predict_utils import parse_output_config


def run_predict(config, run_dir):
    parsed = parse_output_config(config.get("output", {}))
    uncertainty = parsed["uncertainty"]
    unit = parsed["unit"]
    device = str(config.get("model", {}).get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] device={device}")

    if uncertainty == "gt":
        if unit == "image":
            run_fn_csv(config, run_dir)
            return
        if unit == "bbox":
            run_tp_csv(config, run_dir)
            return
        raise ValueError("output.uncertainty='gt' requires output.unit in {'image','bbox'}.")
    if uncertainty == "score":
        run_score_csv(config, run_dir)
        return
    if uncertainty == "meta_detect":
        run_meta_detect_csv(config, run_dir)
        return
    if uncertainty == "mc_dropout":
        run_mc_dropout_csv(config, run_dir)
        return
    if uncertainty == "ensemble":
        run_ensemble_csv(config, run_dir)
        return
    if uncertainty == "full_softmax":
        run_full_softmax_csv(config, run_dir)
        return
    if uncertainty == "energy":
        run_energy_csv(config, run_dir)
        return
    if uncertainty == "entropy":
        run_entropy_csv(config, run_dir)
        return
    if uncertainty == "feature":
        run_feature_csv(config, run_dir)
        return
    if uncertainty == "feature_grad":
        run_feature_grad_csv(config, run_dir)
        return
    if uncertainty == "layer_grad":
        run_layer_grad_csv(config, run_dir)
        return
    raise ValueError(f"Unsupported uncertainty: {uncertainty}")


__all__ = [
    "run_predict",
    "run_fn_csv",
    "run_tp_csv",
    "run_score_csv",
    "run_meta_detect_csv",
    "run_mc_dropout_csv",
    "run_ensemble_csv",
    "run_full_softmax_csv",
    "run_energy_csv",
    "run_entropy_csv",
    "run_feature_csv",
    "run_feature_grad_csv",
    "run_layer_grad_csv",
]

