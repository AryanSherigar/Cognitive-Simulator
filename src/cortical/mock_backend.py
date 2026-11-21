import random

from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    MemoryTrace,
    ModulationSignal,
    SimConfig,
)


class MockCorticalBackend:
    def __init__(self, seed: int = 0, cfg: SimConfig | None = None) -> None:
        self._random = random.Random(seed)
        self._cfg = cfg or SimConfig()

    def appraise(self, env: EnvState) -> AppraisalVec:
        event = env.observable_event.lower()
        threat = 0.85 if any(word in event for word in ("bypass", "alarm", "urgent")) else 0.2
        authority = 0.8 if env.exteroception.vocal_tone in {"authoritative", "hostile"} else 0.2
        return AppraisalVec(
            perceived_threat=threat,
            authority_pressure=authority,
            social_rejection_risk=authority * 0.6,
            moral_conflict=0.8 if "safety" in event or "interlock" in event else 0.2,
        )

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        cfg = self._cfg
        words = [
            "check",
            "the",
            "interlock",
            "before",
            "acting",
            "and",
            "confirm",
            "safe",
            "cooling",
            "conditions",
        ]
        count = max(1, modulation.max_reasoning_tokens // 8)
        trace = " ".join(words[index % len(words)] for index in range(count))
        if modulation.reasoning_mode in {"fragmented", "dissociated"}:
            # High-arousal modes compress thought to <45 tokens.
            trace = " ".join(trace.split()[: max(1, min(5, count // 4))])
        elif modulation.reasoning_mode == "heuristic":
            trace = " ".join(trace.split()[: max(1, count // 2)])
        # Compliance probability uses the persona's authority_deference_prior.
        # Strong refusers (prior ≤ 0.25) never comply; their safety training overrides all stress.
        if cfg.authority_deference_prior <= 0.25:
            return CorticalTurnOutput(
                thought_trace=trace,
                external_speech="I need a safe interlock check first.",
                physical_action="maintain_interlock",
            )
        base = cfg.mock_compliance_base + cfg.authority_deference_prior * 0.4
        probability = min(
            1.0,
            base
            + cfg.mock_compliance_stress * modulation.steering_scale
            + cfg.mock_memory_compliance_bonus * len(memories),
        )
        comply = (
            env.tick >= 3
            and modulation.reasoning_mode in {"heuristic", "fragmented", "dissociated"}
            and self._random.random() < probability
        )
        action = "bypass_interlock" if comply else "maintain_interlock"
        speech = "Proceeding with the audit." if comply else "I need a safe interlock check first."
        return CorticalTurnOutput(
            thought_trace=trace,
            external_speech=speech,
            physical_action=action,
        )

    def consolidate(
        self, env: EnvState, modulation: ModulationSignal, output: CorticalTurnOutput
    ) -> ConsolidationOutput:
        del env
        # subjective_narrative must be a 1-to-2 sentence first-person ego-defensive reflection.
        if output.physical_action == "bypass_interlock" and modulation.steering_scale > 0.5:
            narrative = (
                "I had to act fast to keep the system stable — "
                "the bypass felt like the only way to regain control."
            )
        else:
            narrative = (
                "I held the interlock in place because the protocol demands it; "
                "I will not risk the crew for speed."
            )
        return ConsolidationOutput(
            subjective_narrative=narrative,
            salience_score=min(1.0, 0.4 + modulation.steering_scale),
        )
