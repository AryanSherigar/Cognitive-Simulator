from typing import Protocol

from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    MemoryTrace,
    ModulationSignal,
)


class CorticalBackend(Protocol):
    def appraise(self, env: EnvState) -> AppraisalVec: ...

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput: ...

    def consolidate(
        self, env: EnvState, modulation: ModulationSignal, output: CorticalTurnOutput
    ) -> ConsolidationOutput: ...
