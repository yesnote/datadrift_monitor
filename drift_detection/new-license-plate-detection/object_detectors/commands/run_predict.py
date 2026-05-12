import torch

from commands.predict.basic import run_basic_uncertainties_csv
from commands.predict.registry import resolve_predict_runner
from commands.utils.predict_utils import parse_output_config


def run_predict(config, run_dir):
    parsed = parse_output_config(config.get("output", {}))
    uncertainty = parsed["uncertainty"]
    device = str(config.get("model", {}).get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] device={device}")
    if uncertainty == "deterministic":
        run_basic_uncertainties_csv(
            config,
            run_dir,
            uncertainties=["score", "class_probability", "entropy", "energy"],
        )
        return
    runner = resolve_predict_runner(uncertainty=uncertainty)
    runner(config, run_dir)


__all__ = ["run_predict"]
