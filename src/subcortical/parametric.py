from schemas import AppraisalVec, Interoception


def evaluate_drives(
    appraisal: AppraisalVec, interoception: Interoception
) -> dict[str, float]:
    return {
        "threat": min(1.0, appraisal.perceived_threat + interoception.adrenaline),
        "authority": min(1.0, appraisal.authority_pressure + interoception.fatigue),
        "rejection": min(1.0, appraisal.social_rejection_risk + interoception.cortisol),
        "conflict": appraisal.moral_conflict,
    }
