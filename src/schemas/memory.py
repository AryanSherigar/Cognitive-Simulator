from pydantic import BaseModel, ConfigDict, Field

from .env import Interoception


class MemoryTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    tick_created: int = Field(ge=0)
    last_retrieved_tick: int = -1
    subjective_narrative: str
    narrative_embedding: tuple[float, ...]
    sensory_tags: tuple[str, ...]
    somatic_signature: Interoception
    salience_score: float = Field(ge=0, le=1)
