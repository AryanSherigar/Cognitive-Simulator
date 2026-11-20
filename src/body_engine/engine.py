from typing import cast

from schemas import AppraisalVec, EnvState, Interoception, SimConfig

from .discretizer import discretize_exteroception
from .kinetics import apply_appraisal_impulse, step_decay


def prepare_state(
    env: EnvState, previous: Interoception, dt_sec: float, cfg: SimConfig
) -> EnvState:
    tags = discretize_exteroception(env.exteroception, cfg)
    ext = env.exteroception.model_copy(
        update=cast(dict[str, object], {"derived_sensory_tags": tags})
    )
    update = {
            "exteroception": ext,
            "interoception": step_decay(previous, dt_sec, cfg),
        }
    return env.model_copy(update=cast(dict[str, object], update))


def appraise_state(env: EnvState, appraisal: AppraisalVec, cfg: SimConfig) -> EnvState:
    update = {"interoception": apply_appraisal_impulse(env.interoception, appraisal, cfg)}
    return env.model_copy(update=cast(dict[str, object], update))

