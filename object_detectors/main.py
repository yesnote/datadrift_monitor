import argparse
from pathlib import Path

from dataloaders.dataloader_yolo import load_config
from commands.run_predict import run_predict
from commands.utils.predict_utils import parse_output_config
from commands.utils.run_utils import create_run_dir, save_run_summary, save_used_config

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def _resolve_path_value(raw_path):
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _resolve_config_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_run_dir(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _normalize_config_paths(config):
    model_cfg = config.get("model", {})
    weight_path = model_cfg.get("weights")
    if isinstance(weight_path, str) and weight_path:
        model_cfg["weights"] = str(_resolve_path_value(weight_path))

    dataset_cfg = config.get("dataset", {})
    used_dataset = dataset_cfg.get("used_dataset")
    if isinstance(used_dataset, str):
        names = [used_dataset.strip()]
    elif isinstance(used_dataset, (list, tuple)):
        names = [str(v).strip() for v in used_dataset if str(v).strip()]
    else:
        names = []
    for name in names:
        if name in dataset_cfg:
            active_cfg = dataset_cfg[name]
            root = active_cfg.get("root")
            if isinstance(root, str) and root:
                active_cfg["root"] = str(_resolve_path_value(root))

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="object_detectors/configs/predict_yolov5.yaml")
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _normalize_config_paths(load_config(str(config_path)))
    mode = str(config.get("mode", "")).lower()
    if mode != "predict":
        raise ValueError(f"Unsupported mode: {mode}. Only 'predict' is implemented.")
    parsed_output = parse_output_config(config.get("output", {}))
    uncertainty = str(parsed_output.get("uncertainty", ""))
    target_tag = ""
    save_csv_cfg = config.get("output", {}).get("save_csv", {})
    if uncertainty == "feature_grad":
        raw_vals = save_csv_cfg.get("feature_grad", {}).get("target_value", [])
        raw_list = [str(v).strip().lower() for v in (raw_vals if isinstance(raw_vals, list) else [raw_vals]) if str(v).strip()]
        if raw_list == ["loss"]:
            target_tag = "loss"
        else:
            vals = parsed_output.get("target_values", [])
            target_tag = "-".join([str(v).strip().lower() for v in vals if str(v).strip()])
    elif uncertainty == "layer_grad":
        raw_vals = save_csv_cfg.get("layer_grad", {}).get("target_value", [])
        raw_list = [str(v).strip().lower() for v in (raw_vals if isinstance(raw_vals, list) else [raw_vals]) if str(v).strip()]
        if raw_list == ["loss"]:
            target_tag = "loss"
        else:
            vals = parsed_output.get("layer_target_values", [])
            target_tag = "-".join([str(v).strip().lower() for v in vals if str(v).strip()])

    if args.run_dir:
        run_dir = _resolve_run_dir(args.run_dir)
    else:
        run_dir = create_run_dir(
            uncertainty=parsed_output.get("uncertainty"),
            unit=parsed_output.get("unit"),
            target_value=target_tag,
        ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_used_config(config_path, run_dir)
    run_predict(config, run_dir)
    save_run_summary(run_dir, uncertainty=parsed_output.get("uncertainty", ""), unit=parsed_output.get("unit", ""))


if __name__ == "__main__":
    main()
