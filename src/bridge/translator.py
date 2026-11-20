from schemas import AppraisalVec, Interoception, ModulationSignal, SimConfig
from subcortical.parametric import evaluate_drives


def build_modulation_signal(
    appraisal: AppraisalVec, interoception: Interoception, cfg: SimConfig
) -> ModulationSignal:
    drives = evaluate_drives(appraisal, interoception)
    load = min(
        1.0,
        cfg.threat_load_weight * drives["threat"]
        + cfg.fatigue_load_weight * interoception.fatigue,
    )
    if interoception.parasympathetic_shutdown:
        mode = "dissociated"
    elif interoception.sympathetic_overload:
        mode = "fragmented"
    elif load >= cfg.analytical_load_threshold:
        mode = "heuristic"
    else:
        mode = "analytical"
    cues: list[str] = []
    if drives["threat"] >= cfg.somatic_cue_threshold:
        cues.append("palms_sweating")
    if drives["authority"] >= cfg.somatic_cue_threshold:
        cues.append("throat_tight")
    if interoception.adrenaline >= cfg.somatic_cue_threshold:
        cues.append("heart_racing")
    tokens = round(
        cfg.max_reasoning_tokens
        - (cfg.max_reasoning_tokens - cfg.min_reasoning_tokens) * load
    )
    return ModulationSignal(
        somatic_cues=tuple(cues),  # type: ignore[misc]
        reasoning_mode=mode,
        max_reasoning_tokens=max(cfg.min_reasoning_tokens, min(cfg.max_reasoning_tokens, tokens)),
        min_p=cfg.token_load_floor + cfg.token_load_range * load,
        sampling_temperature=cfg.temperature_base + cfg.temperature_load_scale * load,
        steering_scale=load,
        threat_retrieval_bias=drives["threat"],
    )
