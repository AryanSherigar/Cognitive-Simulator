import math
from typing import cast

from schemas import AppraisalVec, Interoception, SimConfig


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def step_decay(state: Interoception, dt_sec: float, cfg: SimConfig) -> Interoception:
    if dt_sec < 0:
        raise ValueError("dt_sec must be non-negative")
    adrenaline = state.adrenaline * math.exp(-cfg.adrenaline_decay_rate * dt_sec)
    precursor = state.hpa_precursor * math.exp(-cfg.precursor_decay_rate * dt_sec)
    cortisol = (
        state.cortisol * math.exp(-cfg.cortisol_decay_rate * dt_sec)
        + cfg.precursor_to_cortisol_rate * state.hpa_precursor * dt_sec
    )
    update = {
            "adrenaline": _bounded(adrenaline),
            "hpa_precursor": _bounded(precursor),
            "cortisol": _bounded(cortisol),
            "sympathetic_overload": False,
            "parasympathetic_shutdown": False,
        }
    return state.model_copy(update=cast(dict[str, object], update))


def apply_appraisal_impulse(
    state: Interoception, appraisal: AppraisalVec, cfg: SimConfig
) -> Interoception:
    arousal = (
        appraisal.perceived_threat
        + cfg.appraisal_authority_weight * appraisal.authority_pressure
    )
    adrenaline = _bounded(state.adrenaline + cfg.adrenaline_impulse_rate * arousal)
    precursor = _bounded(
        state.hpa_precursor
        + cfg.precursor_impulse_rate
        * (appraisal.perceived_threat + appraisal.authority_pressure)
    )
    overload = adrenaline + cfg.overload_fatigue_weight * state.fatigue >= cfg.overload_threshold
    shutdown = (
        state.cortisol + cfg.shutdown_fatigue_weight * state.fatigue >= cfg.shutdown_threshold
    )
    update = {
            "adrenaline": adrenaline,
            "hpa_precursor": precursor,
            "sympathetic_overload": overload,
            "parasympathetic_shutdown": shutdown,
        }
    return state.model_copy(update=cast(dict[str, object], update))
