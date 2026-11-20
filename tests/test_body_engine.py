from body_engine.discretizer import discretize_exteroception
from body_engine.kinetics import apply_appraisal_impulse, step_decay
from schemas import AppraisalVec, Exteroception, Interoception, SimConfig


def test_discretizer_and_hpa_lag() -> None:
    cfg = SimConfig()
    ext = Exteroception(
        lighting_lux=800,
        ambient_db=88,
        vocal_tone="authoritative",
        interpersonal_distance_m=0.5,
        active_actors=("a", "b", "c"),
    )
    assert "harsh_light" in discretize_exteroception(ext, cfg)
    state = Interoception(adrenaline=0, hpa_precursor=0, cortisol=0, fatigue=0.4, pain=0)
    appraisal = AppraisalVec(
        perceived_threat=1, authority_pressure=1, social_rejection_risk=0, moral_conflict=0
    )
    for _ in range(6):
        state = apply_appraisal_impulse(step_decay(state, 20, cfg), appraisal, cfg)
    assert state.adrenaline > state.cortisol
    assert state.sympathetic_overload
