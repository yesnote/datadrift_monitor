import torch

from commands.predict.registry import resolve_predict_runner
from commands.utils.predict_utils import parse_output_config


def run_predict(config, run_dir):
    parsed = parse_output_config(config.get("output", {}))
    uncertainty = parsed["uncertainty"]
    unit = parsed["unit"]
    device = str(config.get("model", {}).get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] device={device}")
    runner = resolve_predict_runner(uncertainty=uncertainty, unit=unit)
    runner(config, run_dir)


__all__ = ["run_predict"]
