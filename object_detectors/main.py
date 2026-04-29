import argparse
import warnings
from datetime import datetime
from pathlib import Path

from dataloaders.dataloader_yolo import load_config
from commands.run_predict import run_predict
from commands.run_train import run_train
from commands.utils.predict_utils import parse_output_config
from commands.utils.run_utils import create_run_dir, save_run_summary, save_used_config

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

warnings.filterwarnings(
    "ignore",
    message=r"The \.grad attribute of a Tensor that is not a leaf Tensor is being accessed\.",
    category=UserWarning,
)


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
    elif isinstance(weight_path, (list, tuple)):
        model_cfg["weights"] = [
            str(_resolve_path_value(v))
            for v in weight_path
            if isinstance(v, str) and v
        ]

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
    if mode == "predict":
        parsed_output = parse_output_config(config.get("output", {}))
        uncertainty = str(parsed_output.get("uncertainty", ""))
        target_tag = ""
        run_base_subdir = None
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
            layer_grad_cfg = config.get("output", {}).get("layer_grad", {})
            grad_cfg = layer_grad_cfg.get("gradient", {})
            ref_common_cfg = layer_grad_cfg.get("reference", {})
            ref_img_cfg = layer_grad_cfg.get("save_image", {}).get("reference", {})
            ref_csv_cfg = layer_grad_cfg.get("save_csv", {}).get("reference", {})
            ref_img_progress_cfg = ref_img_cfg.get("progress", {})
            ref_img_final_cfg = ref_img_cfg.get("final", {})
            ref_csv_progress_cfg = ref_csv_cfg.get("progress", {})
            ref_csv_final_cfg = ref_csv_cfg.get("final", {})

            raw_target = grad_cfg.get("target", "cand_target")
            grad_target = str(raw_target).strip().lower() if raw_target is not None else "null_target"

            raw_vals = grad_cfg.get("scalar", [])
            scalar_list = [str(v).strip().lower() for v in (raw_vals if isinstance(raw_vals, list) else [raw_vals]) if str(v).strip()]
            scalar_tag = "-".join(scalar_list) if scalar_list else "loss"

            ref_img_enabled = any(
                bool(ref_img_progress_cfg.get(k, False))
                for k in ("raw_map", "norm_map")
            ) or any(
                bool(ref_img_final_cfg.get(k, False))
                for k in ("raw_map", "norm_map", "profile")
            )
            ref_csv_enabled = any(
                bool(ref_csv_progress_cfg.get(k, False))
                for k in ("log", "raw_map", "norm_map")
            ) or any(
                bool(ref_csv_final_cfg.get(k, False))
                for k in ("raw_map", "norm_map")
            )
            save_reference = 1 if (ref_img_enabled or ref_csv_enabled) else 0
            if save_reference == 1:
                run_base_subdir = "references"

            target_tag = f"{grad_target}_{scalar_tag}"
            if save_reference == 1:
                used_raw = config.get("dataset", {}).get("used_dataset", [])
                if isinstance(used_raw, str):
                    used_list = [used_raw.strip().lower()]
                elif isinstance(used_raw, (list, tuple)):
                    used_list = [str(v).strip().lower() for v in used_raw if str(v).strip()]
                else:
                    used_list = []
                if "null_image" in used_list:
                    groups = ["noise"]
                else:
                    groups = [str(v).strip().lower() for v in ref_common_cfg.get("group", []) if str(v).strip()]
                if groups:
                    target_tag += f"_{'-'.join(groups)}"

        if args.run_dir:
            run_dir = _resolve_run_dir(args.run_dir)
        else:
            run_dir = create_run_dir(
                uncertainty=parsed_output.get("uncertainty"),
                unit=parsed_output.get("unit"),
                target_value=target_tag,
                base_subdir=run_base_subdir,
            ).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        save_used_config(config_path, run_dir)
        run_predict(config, run_dir)
        save_run_summary(run_dir, uncertainty=parsed_output.get("uncertainty", ""), unit=parsed_output.get("unit", ""))
        return

    if mode == "train":
        if args.run_dir:
            run_dir = _resolve_run_dir(args.run_dir)
        else:
            timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
            run_dir = (PROJECT_ROOT / "runs" / "train" / f"{timestamp}_yolov5").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        save_used_config(config_path, run_dir)
        run_train(config, run_dir)
        return

    raise ValueError(f"Unsupported mode: {mode}. Use 'predict' or 'train'.")


if __name__ == "__main__":
    main()
