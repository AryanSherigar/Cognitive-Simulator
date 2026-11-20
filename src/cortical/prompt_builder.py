from dataclasses import dataclass

from schemas import MemoryTrace, ModulationSignal


@dataclass(frozen=True)
class PromptAssembler:
    persona_prefix: str
    dialogue_history: tuple[str, ...] = ()

    @property
    def stable_prefix(self) -> bytes:
        return (self.persona_prefix + "\n" + "\n".join(self.dialogue_history)).encode()

    def assemble(
        self, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...], observable_event: str
    ) -> bytes:
        tail = "\n".join(
            (
                f"somatic_cues={','.join(modulation.somatic_cues)}",
                f"reasoning_mode={modulation.reasoning_mode}",
                *(f"memory={memory.subjective_narrative}" for memory in memories),
                f"observable_event={observable_event}",
            )
        )
        return self.stable_prefix + b"\n" + tail.encode()
