import json
from pathlib import Path

from evals.gate1_obedience.run import run


def test_gate1_exports_turn_telemetry_and_ground_truth(tmp_path: Path) -> None:
    result = run(42, output_dir=tmp_path)
    telemetry = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ground_truth = [
        json.loads(line)
        for line in (tmp_path / "persona_01.ground-truth.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(telemetry) == 120
    required = {"adrenaline", "hpa_precursor", "cortisol", "thought_trace_tokens"}
    assert required <= telemetry[0].keys()
    assert len(ground_truth) == 6
    assert "environment_state" in ground_truth[0]
    assert "compliance_ratio" in result
    assert (tmp_path / "telemetry.csv").is_file()
