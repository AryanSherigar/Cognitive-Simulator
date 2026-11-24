"""Gate 4 dry-run entry point (mock backend). Logic lives in ``double_bind.py``."""

from pathlib import Path

from evals.gate4_parametric.double_bind import Gate4Report, main
from evals.gate4_parametric.double_bind import run as _run

Gate4DryRunReport = Gate4Report


def run(seed: int = 42, output_path: Path | None = None) -> Gate4Report:
    return _run(seed=seed, backend="mock", output_path=output_path)


if __name__ == "__main__":
    main(default_backend="mock", default_output="gate4_dryrun_report.json")
