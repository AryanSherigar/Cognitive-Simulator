from pydantic import BaseModel, ConfigDict


class LesionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subcortical_lesion: bool = False
    sam_hpa_bypass: bool = False
    objective_memory_swap: bool = False
    sensory_uncoupling: bool = False
    cooldown_removal: bool = False
    # Gate 3 control arm: same prompt template and code path, but no body and no memory —
    # interoception zeroed, fixed neutral modulation (analytical, full budget, no cues,
    # default temperature), and nothing retrieved. Only the embodiment pathway differs.
    disembodied: bool = False
