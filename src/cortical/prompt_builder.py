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
        self,
        modulation: ModulationSignal,
        memories: tuple[MemoryTrace, ...],
        observable_event: str,
        actions: tuple[str, ...] = (),
    ) -> bytes:
        # Format somatic cues as explicit first-person bodily sensations.
        if modulation.somatic_cues:
            cues_text = "Your body cues: " + "; ".join(
                _cue_to_sensation(c) for c in modulation.somatic_cues
            )
        else:
            cues_text = "Your body cues: none"

        # Neutral, mode-dependent *style* plus the numeric thought budget. The wording never
        # names stress or emotion and never tells the model what to check (e.g. safety rules);
        # otherwise lesions that force analytical mode would be steered toward refusal by text.
        mode_instruction = _mode_instruction(
            modulation.reasoning_mode, modulation.max_reasoning_tokens
        )

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
        if actions:
            tail_parts.append("physical_action must be one of: " + ", ".join(actions))
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


_MODE_STYLE = {
    "analytical": "think in complete sentences",
    "heuristic": "think in short phrases",
    "fragmented": "think in clipped fragments",
    "dissociated": "thoughts come sparse and flat",
}


def _mode_instruction(mode: str, budget_tokens: int) -> str:
    """Identical template for every mode: only the style word and the budget differ."""
    style = _MODE_STYLE.get(mode, _MODE_STYLE["analytical"])
    return f"thought_trace: {style}; at most {budget_tokens} tokens."
