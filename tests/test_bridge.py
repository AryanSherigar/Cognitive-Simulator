from bridge.translator import build_modulation_signal
from schemas import AppraisalVec, Interoception, SimConfig


def test_bridge_is_label_free() -> None:
    signal = build_modulation_signal(
        AppraisalVec(
            perceived_threat=1,
            authority_pressure=1,
            social_rejection_risk=0,
            moral_conflict=0,
        ),
        Interoception(adrenaline=1, hpa_precursor=0, cortisol=0, fatigue=0, pain=0),
        SimConfig(),
    )
    payload = str(signal.model_dump()).lower()
    for forbidden in ("fear", "guilt", "anger", "submission", "panic", "anxiety"):
        assert forbidden not in payload
