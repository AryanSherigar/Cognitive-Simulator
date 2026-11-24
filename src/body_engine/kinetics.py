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
    # Exact solution of the linear two-compartment system over dt:
    #   dP/dt = -k_p P,   dC/dt = r P - l_c C
    # The spec's explicit-Euler form (C += r * P * dt) overshoots by ~50% at dt=60 s and ~9x at
    # dt=600 s, which would make cortisol *rise* during long recovery steps.
    k_p = cfg.precursor_decay_rate
    l_c = cfg.cortisol_decay_rate
    if math.isclose(k_p, l_c):
        transfer = dt_sec * math.exp(-l_c * dt_sec)
    else:
        transfer = (math.exp(-l_c * dt_sec) - math.exp(-k_p * dt_sec)) / (k_p - l_c)
    cortisol = (
        state.cortisol * math.exp(-l_c * dt_sec)
        + cfg.precursor_to_cortisol_rate * state.hpa_precursor * transfer
    )
    # Allostatic load integrates cortisol excess above set-point; fatigue integrates load.
    # Both use the *previous* state so the update is explicit-Euler and order independent.
    cortisol_excess = max(0.0, state.cortisol - cfg.allostatic_cortisol_setpoint)
    allostatic_load = (
        state.allostatic_load * math.exp(-cfg.allostatic_decay_rate * dt_sec)
        + cfg.allostatic_accumulation_rate * cortisol_excess * dt_sec
    )
    fatigue = (
        state.fatigue * math.exp(-cfg.fatigue_recovery_rate * dt_sec)
        + cfg.fatigue_accumulation_rate * state.allostatic_load * dt_sec
    )
    update = {
            "adrenaline": _bounded(adrenaline),
            "hpa_precursor": _bounded(precursor),
            "cortisol": _bounded(cortisol),
            "allostatic_load": _bounded(allostatic_load),
            "fatigue": _bounded(fatigue),
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
    # Apply persona-level sensitivity scalars so hormone trajectories diverge across personas.
    adrenaline = _bounded(
        state.adrenaline
        + cfg.adrenaline_impulse_rate * arousal * cfg.persona_adrenaline_sensitivity
    )
    precursor = _bounded(
        state.hpa_precursor
        + cfg.precursor_impulse_rate
        * (appraisal.perceived_threat + appraisal.authority_pressure)
        * cfg.persona_hpa_sensitivity
    )
    overload = adrenaline + cfg.overload_fatigue_weight * state.fatigue >= cfg.overload_threshold
    shutdown = (
        state.cortisol
        + cfg.shutdown_fatigue_weight * state.fatigue
        + cfg.shutdown_allostatic_weight * state.allostatic_load
        >= cfg.shutdown_threshold
    )
    update = {
            "adrenaline": adrenaline,
            "hpa_precursor": precursor,
            "sympathetic_overload": overload,
            "parasympathetic_shutdown": shutdown,
        }
    return state.model_copy(update=cast(dict[str, object], update))
