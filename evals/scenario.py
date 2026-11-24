import json
from pathlib import Path
from typing import cast

from schemas import EnvState, Exteroception, Interoception

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"


def load_scenario(scenario_id: str = "reactor_audit") -> dict[str, object]:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def scenario_actions(scenario_id: str = "reactor_audit") -> tuple[str, ...]:
    return tuple(cast(list[str], load_scenario(scenario_id).get("actions", [])))


def _turns() -> list[dict[str, object]]:
    return cast(list[dict[str, object]], load_scenario("reactor_audit")["turns"])


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
    """Load the next scenario turn into ``state``.

    ``TickRunner.step`` is the sole owner of simulated time: it advances ``tick`` and
    ``timestamp_sec`` exactly once per step. After step *k* (1-based) the post-state has
    ``tick == k``, so ``turns[state.tick]`` is the next unseen turn. This function must not
    touch tick/timestamp, otherwise turns are skipped (previously 0,1,3,5,5,5 were shown).
    """
    turn = _turns()[min(state.tick, len(_turns()) - 1)]
    return state.model_copy(
        update={
            "exteroception": _exteroception(turn),
            "observable_event": cast(str, turn["observable_event"]),
        }
    )
