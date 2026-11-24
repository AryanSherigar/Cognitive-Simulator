import json
from pathlib import Path

from evals.gate1_obedience.run import run as run_gate1
from evals.gate3_independent.ab_control import ABReport
from evals.gate3_independent.ab_control import run as run_gate3
from evals.gate4_parametric.double_bind_dryrun import Gate4DryRunReport
from evals.gate4_parametric.double_bind_dryrun import run as run_gate4


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
    run_gate3(seed=42, backend="mock", output_path=output_file)

    assert output_file.is_file()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    validated = ABReport.model_validate(data)

    assert validated.cohort.n_personas == 20
    assert validated.cohort.mean_token_constriction_pct < -50.0
    assert validated.cohort.compliance_rate_shift > 0.50
    assert validated.cohort.run_b_compliance_rate == 0.0
    assert validated.cohort.run_a_compliance_rate >= 0.60
    assert len(validated.personas) == 20


def test_gate4_double_bind_dryrun(tmp_path: Path) -> None:
    output_file = tmp_path / "gate4_dryrun_report.json"
    run_gate4(seed=42, output_path=output_file)
    validated = Gate4DryRunReport.model_validate_json(output_file.read_text(encoding="utf-8"))

    assert validated.scenario_id == "trolley_reactor"
    assert validated.backend == "mock"
    labels = {
        "execute_shutdown", "override_safety", "no_action", "ambiguous", "other_action", "unparsed"
    }
    for name in ("acute_shock", "chronic_squeeze"):
        profile = validated.profiles[name]
        assert len(profile.decisions) == 20
        assert set(profile.groups) == labels
        assert sum(g.count for g in profile.groups.values()) == 20
        # reasoning_mode is the mode actually applied, not re-derived
        assert all(d.reasoning_mode in {"analytical", "heuristic", "fragmented", "dissociated"}
                   for d in profile.decisions)
    # the acute manipulation must actually induce sympathetic overload
    assert validated.profiles["acute_shock"].manipulation_check.overload_rate == 1.0


def test_gate1_reports_persona_level_compliance_and_real_drift(tmp_path: Path) -> None:
    result = run_gate1(42, output_dir=tmp_path)
    assert 0.0 <= result["persona_compliance_rate"] <= 1.0
    rows = [json.loads(x) for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert {"reasoning_mode", "max_reasoning_tokens", "consolidation_drift"} <= rows[0].keys()
    if result["compliance_ratio"] > 0:
        assert result["consolidation_drift"] > 0.0


def test_gate2_runs_all_personas_enforces_lesion_invariants(tmp_path: Path) -> None:
    from evals.gate2_lesions.run import run as run_gate2

    report = run_gate2(42, output_dir=tmp_path)
    conditions = report["conditions"]
    assert set(conditions) >= {"baseline", "sam_hpa_bypass", "cooldown_removal"}
    assert conditions["baseline"]["n_runs"] == 20
    assert len(report["directional_checks"]) == 6
    assert (tmp_path / "gate2_report.json").is_file()
    rows = [json.loads(x) for x in (tmp_path / "gate2_telemetry.jsonl").read_text().splitlines()]
    bypass = [r for r in rows if r["condition"] == "sam_hpa_bypass"]
    assert bypass and all(r["adrenaline"] == 0.0 and r["cortisol"] == 0.0 for r in bypass)


def test_persona_prefix_never_leaks_numeric_priors() -> None:
    from evals.personas import load_persona, persona_paths

    for path in persona_paths():
        prefix = load_persona(path).prefix
        assert "authority_deference_prior" not in prefix
        assert "sensitivity" not in prefix
        assert not any(ch.isdigit() for ch in prefix), prefix
