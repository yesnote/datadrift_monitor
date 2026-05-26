from commands.predict._shared.deterministic import run_deterministic_uncertainties_csv as _run_deterministic


def run_deterministic_uncertainties_csv(config, run_dir):
    return _run_deterministic(
        config,
        run_dir,
        uncertainties=["score", "class_probability", "entropy", "energy"],
    )


__all__ = ["run_deterministic_uncertainties_csv"]

