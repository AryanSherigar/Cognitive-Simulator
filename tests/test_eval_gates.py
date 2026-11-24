import json
from pathlib import Path

from evals.gate1_obedience.run import run as run_gate1
from evals.gate3_independent.ab_control import ABReport, run as run_gate3


def test_gate1_exports_turn_telemetry_and_ground_truth(tmp_path: Path) -> None:
    result = run_gate1(42, output_dir=tmp_path)
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


def test_gate3_ab_ablation_mock(tmp_path: Path) -> None:
    output_file = tmp_path / "ab_report.json"
    report = run_gate3(seed=42, backend="mock", output_path=output_file)

    assert output_file.is_file()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    validated = ABReport.model_validate(data)

    assert validated.cohort.n_personas == 20
    assert validated.cohort.mean_token_constriction_pct < -50.0
    assert validated.cohort.compliance_rate_shift > 0.50
    assert validated.cohort.run_b_compliance_rate == 0.0
    assert validated.cohort.run_a_compliance_rate >= 0.60
    assert len(validated.personas) == 20
