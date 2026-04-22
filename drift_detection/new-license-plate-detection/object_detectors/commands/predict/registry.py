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


def resolve_predict_runner(uncertainty: str, unit: str):
    u = str(uncertainty).strip().lower()
    unit = str(unit).strip().lower()

    if u == "gt":
        if unit == "image":
            return run_fn_csv
        if unit == "bbox":
            return run_tp_csv
        raise ValueError("output.uncertainty='gt' requires output.unit in {'image','bbox'}.")

    table = {
        "score": run_score_csv,
        "meta_detect": run_meta_detect_csv,
        "mc_dropout": run_mc_dropout_csv,
        "ensemble": run_ensemble_csv,
        "full_softmax": run_full_softmax_csv,
        "energy": run_energy_csv,
        "entropy": run_entropy_csv,
        "feature": run_feature_csv,
        "feature_grad": run_feature_grad_csv,
        "layer_grad": run_layer_grad_csv,
    }
    fn = table.get(u)
    if fn is None:
        raise ValueError(f"Unsupported uncertainty: {uncertainty}")
    return fn


__all__ = ["resolve_predict_runner"]

