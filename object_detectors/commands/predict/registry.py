from commands.predict import (
    run_energy_csv,
    run_ensemble_csv,
    run_entropy_csv,
    run_class_probability_csv,
    run_layer_grad_csv,
    run_mc_dropout_csv,
    run_meta_detect_csv,
    run_predict_dump_csv,
    run_score_csv,
    run_tp_csv,
)


def resolve_predict_runner(uncertainty: str):
    u = str(uncertainty).strip().lower()

    if u == "gt":
        return run_tp_csv

    table = {
        "score": run_score_csv,
        "meta_detect": run_meta_detect_csv,
        "mc_dropout": run_mc_dropout_csv,
        "ensemble": run_ensemble_csv,
        "class_probability": run_class_probability_csv,
        "energy": run_energy_csv,
        "entropy": run_entropy_csv,
        "layer_grad": run_layer_grad_csv,
        "predict_dump": run_predict_dump_csv,
    }
    fn = table.get(u)
    if fn is None:
        raise ValueError(f"Unsupported uncertainty: {uncertainty}")
    return fn


__all__ = ["resolve_predict_runner"]

