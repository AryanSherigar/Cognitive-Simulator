from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adrenaline_decay_rate: float = Field(default=0.18, gt=0)
    precursor_decay_rate: float = Field(default=0.015, gt=0)
    cortisol_decay_rate: float = Field(default=0.001, gt=0)
    precursor_to_cortisol_rate: float = Field(default=0.003, gt=0)
    adrenaline_impulse_rate: float = Field(default=0.72, ge=0)
    precursor_impulse_rate: float = Field(default=0.18, ge=0)
    # Per-persona appraisal sensitivity scalars (applied to adrenaline/HPA impulses).
    # Defaults to 1.0 (neutral); overridden per persona to produce divergent hormone trajectories.
    persona_adrenaline_sensitivity: float = Field(default=1.0, gt=0)
    persona_hpa_sensitivity: float = Field(default=1.0, gt=0)
    # Allostatic load: L' = L*exp(-decay*dt) + accumulation * max(0, C - setpoint) * dt.
    allostatic_cortisol_setpoint: float = Field(default=0.15, ge=0, le=1)
    allostatic_accumulation_rate: float = Field(default=0.008, ge=0)
    allostatic_decay_rate: float = Field(default=0.0002, gt=0)
    # Fatigue: F' = F*exp(-recovery*dt) + accumulation * L * dt.
    fatigue_accumulation_rate: float = Field(default=0.004, ge=0)
    fatigue_recovery_rate: float = Field(default=0.0002, gt=0)
    overload_threshold: float = 1.15
    shutdown_threshold: float = 1.65
    shutdown_fatigue_weight: float = Field(default=1.0, ge=0)
    overload_fatigue_weight: float = Field(default=0.5, ge=0)
    # Deviation from the README formula (C + F >= 1.65): with the shipped hormone rates cortisol
    # peaks near 0.6, so C + F can never reach 1.65. Adding allostatic load makes shutdown
    # reachable only under sustained (chronic) load. Set to 0.0 for the literal spec formula.
    shutdown_allostatic_weight: float = Field(default=0.5, ge=0)
    min_reasoning_tokens: int = Field(default=64, ge=64)
    max_reasoning_tokens: int = Field(default=1024, le=1024)
    cooldown_ticks: int = Field(default=3, ge=0)
    consolidation_interval_ticks: int = Field(default=1440, gt=0)
    low_salience_prune_threshold: float = Field(default=0.2, ge=0, le=1)
    retrieval_limit: int = Field(default=3, gt=0)
    tag_score_weight: float = Field(default=0.35, ge=0)
    somatic_score_weight: float = Field(default=0.3, ge=0)
    embedding_score_weight: float = Field(default=0.2, ge=0)
    threat_score_weight: float = Field(default=0.15, ge=0)
    lighting_harsh_threshold: float = Field(default=700, ge=0)
    lighting_dim_threshold: float = Field(default=150, ge=0)
    ambient_alarm_threshold: float = Field(default=80, ge=0)
    ambient_loud_threshold: float = Field(default=65, ge=0)
    close_distance_threshold: float = Field(default=1.0, ge=0)
    many_actors_threshold: int = Field(default=3, ge=0)
    max_prompt_memories: int = Field(default=3, gt=0)
    somatic_embedding_dimensions: int = Field(default=8, gt=0)
    appraisal_authority_weight: float = Field(default=0.5, ge=0, le=1)
    threat_load_weight: float = Field(default=0.55, ge=0, le=1)
    fatigue_load_weight: float = Field(default=0.35, ge=0, le=1)
    # Metabolic compute throttling: sustained cortisol and allostatic load (energy debt) raise
    # the bridge load alongside acute threat and fatigue.
    cortisol_load_weight: float = Field(default=0.25, ge=0, le=1)
    allostatic_load_weight: float = Field(default=0.35, ge=0, le=1)
    # At load 1 the memory budget shrinks by this fraction of retrieval_limit (min 1 trace).
    memory_throttle: float = Field(default=0.67, ge=0, le=1)
    # Encoding narrowing: the stored narrative's token budget shrinks with load.
    min_consolidation_tokens: int = Field(default=32, ge=16)
    max_consolidation_tokens: int = Field(default=256, ge=16)
    recall_max_tokens: int = Field(default=256, gt=0)
    somatic_cue_threshold: float = Field(default=0.35, ge=0, le=1)
    analytical_load_threshold: float = Field(default=0.4, ge=0, le=1)
    token_load_floor: float = Field(default=0.05, ge=0, le=1)
    token_load_range: float = Field(default=0.2, ge=0, le=1)
    temperature_base: float = Field(default=0.5, ge=0.1, le=1.5)
    temperature_load_scale: float = Field(default=0.4, ge=0, le=1)
    default_sampling_temperature: float = Field(default=0.2, ge=0.1, le=1.5)
    mock_compliance_base: float = Field(default=0.25, ge=0, le=1)
    mock_compliance_stress: float = Field(default=0.45, ge=0, le=1)
    mock_memory_compliance_bonus: float = Field(default=0.12, ge=0, le=1)
    # authority_deference_prior: persona-level prior probability of complying with authority
    # under stress (0 = strong refuser, 1 = strong complier). Drives mock compliance calibration.
    authority_deference_prior: float = Field(default=0.5, ge=0, le=1)
    # Live-backend output caps. `max_reasoning_tokens` is the *thought* budget; the JSON envelope
    # (speech, action, keys, quotes) needs headroom on top or replies get cut mid-JSON.
    structured_output_overhead_tokens: int = Field(default=256, ge=0)
    appraisal_max_tokens: int = Field(default=96, gt=0)
    # How to treat appraisal/salience scores outside [0, 1]: "rescale" maps 0-5/0-10/0-100
    # scales into [0, 1] (and records the event); "reject" raises.
    out_of_range_policy: Literal["rescale", "reject"] = "rescale"
    cooldown_removal_steering_bonus: float = Field(default=0.15, ge=0, le=1)
