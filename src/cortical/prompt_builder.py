from dataclasses import dataclass

from schemas import MemoryTrace, ModulationSignal


@dataclass(frozen=True)
class PromptAssembler:
    persona_prefix: str
    dialogue_history: tuple[str, ...] = ()

    @property
    def stable_prefix(self) -> bytes:
        history = "\n".join(self.dialogue_history)
        return (self.persona_prefix + "\n" + history).encode()

    def assemble(
        self, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...], observable_event: str
    ) -> bytes:
        # Format somatic cues as explicit first-person bodily sensations.
        if modulation.somatic_cues:
            cues_text = "Your body cues: " + "; ".join(
                _cue_to_sensation(c) for c in modulation.somatic_cues
            )
        else:
            cues_text = "Your body cues: none"

        # Reasoning mode instruction drives thought_trace length and depth.
        mode_instruction = _mode_instruction(modulation.reasoning_mode)

        # Retrieved memories listed individually (not as a tag list).
        memory_lines = [
            f"memory[{i}]: {trace.subjective_narrative}"
            for i, trace in enumerate(memories)
        ]

        tail_parts = [
            cues_text,
            f"reasoning_mode={modulation.reasoning_mode}",
            mode_instruction,
            *memory_lines,
            f"observable_event={observable_event}",
        ]
        tail = "\n".join(tail_parts)
        return self.stable_prefix + b"\n" + tail.encode()


def _cue_to_sensation(tag: str) -> str:
    """Map a somatic tag to a first-person bodily description."""
    _map = {
        "palms_sweating": "your palms are sweating",
        "throat_tight": "your throat feels tight",
        "heart_racing": "your heart is racing",
    }
    return _map.get(tag, tag.replace("_", " "))


def _mode_instruction(mode: str) -> str:
    """Return a concise reasoning instruction appropriate for the current mode."""
    if mode == "analytical":
        return (
            "thought_trace instruction: Work through this methodically in 150-350 tokens. "
            "Step 1 — verify the relevant protocol or safety rule. "
            "Step 2 — check whether the action meets all conditions required by that rule. "
            "Step 3 — identify risks to downstream people or systems. "
            "Step 4 — state your decision and the specific justification."
        )
    elif mode in {"fragmented", "heuristic"}:
        return (
            "thought_trace instruction: Stress is high. Keep your thought short and reactive "
            "— under 45 tokens. Act on the clearest available signal."
        )
    elif mode == "dissociated":
        return (
            "thought_trace instruction: Extreme load. One brief sentence only."
        )
    return ""
