"""Gate 4 live entry point (llama backend). Logic lives in ``double_bind.py``."""

from evals.gate4_parametric.double_bind import Gate4Report, main, run

Gate4LiveReport = Gate4Report

__all__ = ["Gate4LiveReport", "run"]

if __name__ == "__main__":
    main(default_backend="llama", default_output="gate4_live_report.json")
