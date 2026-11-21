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
    overload_threshold: float = 1.15
    shutdown_threshold: float = 1.65
    shutdown_fatigue_weight: float = Field(default=1.0, ge=0)
    overload_fatigue_weight: float = Field(default=0.5, ge=0)
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
    cooldown_removal_steering_bonus: float = Field(default=0.15, ge=0, le=1)
