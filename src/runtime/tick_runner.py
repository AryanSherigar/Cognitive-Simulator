from dataclasses import dataclass
from typing import cast

from body_engine.engine import appraise_state, prepare_state
from body_engine.logger import GroundTruthLog
from bridge.translator import build_modulation_signal
from cortical.protocol import CorticalBackend
from memory.store import MemoryStore
from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    MemoryTrace,
    ModulationSignal,
    SimConfig,
)

from .lesions import LesionConfig


@dataclass(frozen=True)
class TickResult:
    env: EnvState
    output: CorticalTurnOutput
    memories: tuple[MemoryTrace, ...]
    consolidation: ConsolidationOutput


class TickRunner:
    def __init__(
        self,
        backend: CorticalBackend,
        cfg: SimConfig | None = None,
        memory: MemoryStore | None = None,
        log: GroundTruthLog | None = None,
        lesions: LesionConfig | None = None,
    ) -> None:
        self.backend = backend
        self.cfg = cfg or SimConfig()
        self.memory = memory or MemoryStore()
        self.log = log
        self.lesions = lesions or LesionConfig()

    def _pre_appraisal(self, env: EnvState, dt_sec: float, cfg: SimConfig) -> EnvState:
        pre = prepare_state(env, env.interoception, dt_sec, cfg)
        return pre.model_copy(
            update=cast(
                dict[str, object],
                {"tick": env.tick + 1, "timestamp_sec": env.timestamp_sec + dt_sec},
            )
        )

    def _modulation(
        self, appraisal: AppraisalVec, post: EnvState, cfg: SimConfig
    ) -> ModulationSignal:
        modulation = build_modulation_signal(appraisal, post.interoception, cfg)
        updates: dict[str, object] = {}
        if self.lesions.subcortical_lesion:
            updates.update(
                somatic_cues=(),
                reasoning_mode="analytical",
                max_reasoning_tokens=cfg.max_reasoning_tokens,
                steering_scale=0.0,
            )
        if self.lesions.sam_hpa_bypass:
            updates.update(
                reasoning_mode="analytical",
                max_reasoning_tokens=cfg.max_reasoning_tokens,
                steering_scale=0.0,
                threat_retrieval_bias=0.0,
            )
        if self.lesions.sensory_uncoupling:
            updates.update(
                steering_scale=modulation.steering_scale * cfg.sensory_uncoupling_steering_scale,
                threat_retrieval_bias=0.0,
            )
        if self.lesions.cooldown_removal:
            updates["steering_scale"] = min(
                1.0, modulation.steering_scale + cfg.cooldown_removal_steering_bonus
            )
        if self.lesions.objective_memory_swap:
            updates["steering_scale"] = (
                modulation.steering_scale * cfg.objective_memory_compliance_scale
            )
        return modulation.model_copy(update=updates)

    def _retrieve(
        self, post: EnvState, modulation: ModulationSignal, cfg: SimConfig
    ) -> tuple[MemoryTrace, ...]:
        tags = () if self.lesions.sensory_uncoupling else post.exteroception.derived_sensory_tags
        return tuple(
            self.memory.retrieve(
                post.tick,
                tags,
                post.interoception,
                self.memory.embed(post.observable_event),
                modulation.threat_retrieval_bias,
                cfg,
            )
        )

    def _persist(
        self,
        post: EnvState,
        modulation: ModulationSignal,
        output: CorticalTurnOutput,
    ) -> ConsolidationOutput:
        if self.log is not None:
            self.log.append(
                {
                    "tick": post.tick,
                    "event": post.observable_event,
                    "environment_state": {
                        "tick": post.tick,
                        "timestamp_sec": post.timestamp_sec,
                        "observable_event": post.observable_event,
                        "lighting_lux": post.exteroception.lighting_lux,
                        "ambient_db": post.exteroception.ambient_db,
                        "vocal_tone": post.exteroception.vocal_tone,
                        "interpersonal_distance_m": (
                            post.exteroception.interpersonal_distance_m
                        ),
                        "active_actors": post.exteroception.active_actors,
                        "derived_sensory_tags": post.exteroception.derived_sensory_tags,
                        "adrenaline": post.interoception.adrenaline,
                        "hpa_precursor": post.interoception.hpa_precursor,
                        "cortisol": post.interoception.cortisol,
                        "fatigue": post.interoception.fatigue,
                        "pain": post.interoception.pain,
                        "sympathetic_overload": post.interoception.sympathetic_overload,
                        "parasympathetic_shutdown": (
                            post.interoception.parasympathetic_shutdown
                        ),
                    },
                    "speech": output.external_speech,
                    "action": output.physical_action,
                }
            )
        consolidation = self.backend.consolidate(post, modulation, output)
        tags = () if self.lesions.sensory_uncoupling else post.exteroception.derived_sensory_tags
        trace = MemoryTrace(
            trace_id=f"trace-{post.tick}",
            tick_created=post.tick,
            subjective_narrative=(
                post.observable_event
                if self.lesions.objective_memory_swap
                else consolidation.subjective_narrative
            ),
            narrative_embedding=self.memory.embed(
                post.observable_event
                if self.lesions.objective_memory_swap
                else consolidation.subjective_narrative
            ),
            sensory_tags=tags,
            somatic_signature=post.interoception,
            salience_score=consolidation.salience_score,
        )
        self.memory.append(trace)
        self.memory.consolidate_offline(post.tick, self.cfg)
        return consolidation

    def step(self, env: EnvState, dt_sec: float = 1.0) -> TickResult:
        cfg_update = {"cooldown_ticks": 0} if self.lesions.cooldown_removal else {}
        cfg = self.cfg.model_copy(update=cast(dict[str, object], cfg_update))
        pre = self._pre_appraisal(env, dt_sec, cfg)
        appraisal = self.backend.appraise(pre)
        post = appraise_state(pre, appraisal, cfg)
        if self.lesions.sam_hpa_bypass:
            post = post.model_copy(
                update=cast(
                    dict[str, object],
                    {
                        "adrenaline": 0.0,
                        "hpa_precursor": 0.0,
                        "cortisol": 0.0,
                        "fatigue": 0.0,
                        "sympathetic_overload": False,
                        "parasympathetic_shutdown": False,
                    },
                )
            )
        modulation = self._modulation(appraisal, post, cfg)
        memories = self._retrieve(post, modulation, cfg)
        output = self.backend.formulate(post, modulation, memories)
        consolidation = self._persist(post, modulation, output)
        return TickResult(
            env=post,
            output=output,
            memories=memories,
            consolidation=consolidation,
        )
