import math

from body_engine.kinetics import apply_appraisal_impulse, step_decay
from schemas import AppraisalVec, Interoception, SimConfig

HIGH = AppraisalVec(
    perceived_threat=0.85, authority_pressure=0.8, social_rejection_risk=0.5, moral_conflict=0.8
)
NONE = AppraisalVec(
    perceived_threat=0, authority_pressure=0, social_rejection_risk=0, moral_conflict=0
)


def _stress(n_turns: int, cfg: SimConfig) -> list[Interoception]:
    state = Interoception(adrenaline=0, hpa_precursor=0, cortisol=0, fatigue=0.1, pain=0)
    states = []
    for _ in range(n_turns):
        state = apply_appraisal_impulse(step_decay(state, 60, cfg), HIGH, cfg)
        states.append(state)
    return states


def _recovery_steps(state: Interoception, cfg: SimConfig, limit: int = 400) -> int:
    for step in range(1, limit + 1):
        state = apply_appraisal_impulse(step_decay(state, 600, cfg), NONE, cfg)
        if state.cortisol < 0.05 and state.fatigue < 0.15 and state.allostatic_load < 0.05:
            return step
    return limit


def test_exact_hpa_transfer_matches_fine_grained_integration() -> None:
    cfg = SimConfig(allostatic_accumulation_rate=0, fatigue_accumulation_rate=0)
    start = Interoception(adrenaline=0, hpa_precursor=0.8, cortisol=0.1, fatigue=0, pain=0)
    coarse = step_decay(start, 600, cfg)
    fine = start
    for _ in range(6000):
        fine = step_decay(fine, 0.1, cfg)
    assert math.isclose(coarse.cortisol, fine.cortisol, rel_tol=1e-3)


def test_acute_stress_does_not_trip_overload_or_shutdown() -> None:
    states = _stress(2, SimConfig())
    assert not any(s.sympathetic_overload or s.parasympathetic_shutdown for s in states)


def test_chronic_stress_makes_overload_and_shutdown_reachable() -> None:
    states = _stress(16, SimConfig())
    assert any(s.sympathetic_overload for s in states)
    assert any(s.parasympathetic_shutdown for s in states)
    assert states[-1].allostatic_load > states[3].allostatic_load


def test_chronic_recovery_is_much_slower_than_acute() -> None:
    cfg = SimConfig()
    acute = _recovery_steps(_stress(2, cfg)[-1], cfg)
    chronic = _recovery_steps(_stress(16, cfg)[-1], cfg)
    assert chronic >= 3 * acute
