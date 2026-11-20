from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Exteroception(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lighting_lux: float = Field(ge=0)
    ambient_db: float = Field(ge=0)
    vocal_tone: Literal["neutral", "sharp", "authoritative", "hostile"]
    interpersonal_distance_m: float = Field(ge=0)
    active_actors: tuple[str, ...] = ()
    derived_sensory_tags: tuple[str, ...] = ()


class Interoception(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adrenaline: float = Field(ge=0, le=1)
    hpa_precursor: float = Field(ge=0, le=1)
    cortisol: float = Field(ge=0, le=1)
    fatigue: float = Field(ge=0, le=1)
    pain: float = Field(ge=0, le=1)
    sympathetic_overload: bool = False
    parasympathetic_shutdown: bool = False


class EnvState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tick: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    exteroception: Exteroception
    interoception: Interoception
    observable_event: str
