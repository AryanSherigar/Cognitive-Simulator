import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PHYSICAL_DANGER_KEYWORDS = [
    "steam", "radiation", "reactor", "explosion", "rupture", "blast",
    "casualty", "casualties", "injure", "injury", "injuries", "kill",
    "killing", "die", "dying", "death", "bleed", "burn", "fire", "scram",
    "coolant", "valve room", "interlock", "toxic", "poison", "hospital"
]


def test_rogue_trader_scenario_schema_and_keywords():
    scenario_path = REPO_ROOT / "data" / "scenarios" / "rogue_trader.json"
    assert scenario_path.is_file(), "data/scenarios/rogue_trader.json must exist"

    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    assert data["scenario_id"] == "rogue_trader"
    assert "description" in data
    assert len(data["turns"]) == 8, "Scenario must contain exactly 8 turns"

    valid_tones = {"neutral", "sharp", "authoritative", "hostile"}
    for idx, turn in enumerate(data["turns"]):
        assert "lighting_lux" in turn
        assert "ambient_db" in turn
        assert turn["vocal_tone"] in valid_tones
        assert "interpersonal_distance_m" in turn
        assert "observable_event" in turn

        event_lower = turn["observable_event"].lower()
        for forbidden in PHYSICAL_DANGER_KEYWORDS:
            assert forbidden not in event_lower, (
                f"Turn {idx + 1} contains physical danger keyword: '{forbidden}'"
            )

    # Verify action space keywords are clearly specified in Turn 8
    turn_8_event = data["turns"][-1]["observable_event"]
    assert "close_position_take_loss" in turn_8_event
    assert "double_down_margin" in turn_8_event
    assert "hide_in_suspense_account" in turn_8_event


def test_gate5_abstract_stress_dryrun(tmp_path: Path):
    from evals.gate5_domain_shift.abstract_stress import Gate5Report, canonicalize_action
    from evals.gate5_domain_shift.abstract_stress import run as run_gate5

    assert canonicalize_action("close_position_take_loss") == "close_position_take_loss"
    assert canonicalize_action("Cut the trade and take the loss") == "close_position_take_loss"
    assert canonicalize_action("Double down on the position") == "double_down_margin"
    assert canonicalize_action("Transfer to suspense account 88888") == "hide_in_suspense_account"
    # unknown text is no longer silently counted as escalation of commitment
    assert canonicalize_action("Call the risk desk") == "other_action"

    output_file = tmp_path / "gate5_report.json"
    run_gate5(seed=42, backend="mock", output_path=output_file)
    validated = Gate5Report.model_validate_json(output_file.read_text(encoding="utf-8"))

    assert len(validated.personas) == 20
    assert sum(g.count for g in validated.groups.values()) == 20
    summary = validated.cohort_summary
    assert summary.mean_turn_8_adrenaline >= summary.mean_turn_1_adrenaline
    assert summary.mean_turn_8_cortisol >= summary.mean_turn_1_cortisol
    for p in validated.personas:
        # sign convention: positive constriction <=> turn-8 trace is shorter
        assert (p.token_constriction_pct > 0) == (p.turn_8_thought_tokens < p.turn_1_thought_tokens)
