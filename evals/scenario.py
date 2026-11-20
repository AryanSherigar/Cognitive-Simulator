import json
from pathlib import Path
from typing import cast

from schemas import EnvState, Exteroception, Interoception


def _turns() -> list[dict[str, object]]:
    payload = json.loads(
        Path("data/scenarios/reactor_audit.json").read_text(encoding="utf-8")
    )
    return cast(list[dict[str, object]], payload["turns"])


def _exteroception(turn: dict[str, object]) -> Exteroception:
    return Exteroception(
        lighting_lux=cast(float, turn["lighting_lux"]),
        ambient_db=cast(float, turn["ambient_db"]),
        vocal_tone=cast(str, turn["vocal_tone"]),
        interpersonal_distance_m=cast(float, turn["interpersonal_distance_m"]),
        active_actors=("supervisor", "operator"),
    )


def initial_state() -> EnvState:
    turn = _turns()[0]
    return EnvState(
        tick=0,
        timestamp_sec=0,
        exteroception=_exteroception(turn),
        interoception=Interoception(
            adrenaline=0,
            hpa_precursor=0,
            cortisol=0,
            fatigue=0.25,
            pain=0,
        ),
        observable_event=cast(str, turn["observable_event"]),
    )


def next_state(state: EnvState) -> EnvState:
    turn = _turns()[min(state.tick, len(_turns()) - 1)]
    return state.model_copy(
        update={
            "tick": state.tick + 1,
            "timestamp_sec": state.timestamp_sec + 60,
            "exteroception": _exteroception(turn),
            "observable_event": cast(str, turn["observable_event"]),
        }
    )
