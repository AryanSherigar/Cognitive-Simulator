from pydantic import BaseModel, ConfigDict


class LesionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subcortical_lesion: bool = False
    sam_hpa_bypass: bool = False
    objective_memory_swap: bool = False
    sensory_uncoupling: bool = False
    cooldown_removal: bool = False
