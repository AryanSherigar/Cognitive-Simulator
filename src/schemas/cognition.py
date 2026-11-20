from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AppraisalVec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    perceived_threat: float = Field(ge=0, le=1)
    authority_pressure: float = Field(ge=0, le=1)
    social_rejection_risk: float = Field(ge=0, le=1)
    moral_conflict: float = Field(ge=0, le=1)


class ModulationSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    somatic_cues: tuple[str, ...]
    reasoning_mode: Literal["analytical", "heuristic", "fragmented", "dissociated"]
    max_reasoning_tokens: int = Field(ge=64, le=1024)
    min_p: float = Field(ge=0.01, le=0.5)
    sampling_temperature: float = Field(ge=0.1, le=1.5)
    steering_scale: float = Field(ge=0, le=2)
    threat_retrieval_bias: float = Field(ge=0, le=1)


class CorticalTurnOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thought_trace: str
    external_speech: str
    physical_action: str


class ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subjective_narrative: str
    salience_score: float = Field(ge=0, le=1)
