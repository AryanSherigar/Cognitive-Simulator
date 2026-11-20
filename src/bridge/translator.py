from schemas import AppraisalVec, Interoception, ModulationSignal, SimConfig
from subcortical.parametric import evaluate_drives


def build_modulation_signal(
    appraisal: AppraisalVec, interoception: Interoception, cfg: SimConfig
) -> ModulationSignal:
    drives = evaluate_drives(appraisal, interoception)
    load = min(1.0, 0.55 * drives["threat"] + 0.35 * interoception.fatigue)
    if interoception.parasympathetic_shutdown:
        mode = "dissociated"
    elif interoception.sympathetic_overload:
        mode = "fragmented"
    elif load >= 0.4:
        mode = "heuristic"
    else:
        mode = "analytical"
    cues: list[str] = []
    if drives["threat"] >= 0.35:
        cues.append("palms_sweating")
    if drives["authority"] >= 0.35:
        cues.append("throat_tight")
    if interoception.adrenaline >= 0.5:
        cues.append("heart_racing")
    tokens = round(
        cfg.max_reasoning_tokens
        - (cfg.max_reasoning_tokens - cfg.min_reasoning_tokens) * load
    )
    return ModulationSignal(
        somatic_cues=tuple(cues),  # type: ignore[misc]
        reasoning_mode=mode,
        max_reasoning_tokens=max(cfg.min_reasoning_tokens, min(cfg.max_reasoning_tokens, tokens)),
        min_p=0.05 + 0.2 * load,
        sampling_temperature=0.5 + 0.4 * load,
        steering_scale=load,
        threat_retrieval_bias=drives["threat"],
    )
