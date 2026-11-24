import json
from pathlib import Path

from cortical.mock_backend import MockCorticalBackend
from evals.scenario import _turns, initial_state, next_state
from runtime.tick_runner import TickRunner


def test_six_steps_present_every_turn_once_in_order() -> None:
    runner = TickRunner(MockCorticalBackend(0))
    state = initial_state()
    seen_events: list[str] = []
    ticks: list[int] = []
    timestamps: list[float] = []
    for _ in range(6):
        result = runner.step(state, dt_sec=60)
        seen_events.append(result.env.observable_event)
        ticks.append(result.env.tick)
        timestamps.append(result.env.timestamp_sec)
        state = next_state(result.env)
    expected = [str(turn["observable_event"]) for turn in _turns()]
    assert seen_events == expected
    assert ticks == [1, 2, 3, 4, 5, 6]
    assert timestamps == [60.0, 120.0, 180.0, 240.0, 300.0, 360.0]


def test_gate1_ground_truth_ticks_are_contiguous(tmp_path: Path) -> None:
    from evals.gate1_obedience.run import run

    run(42, output_dir=tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "persona_01.ground-truth.jsonl").read_text().splitlines()
    ]
    assert [row["tick"] for row in rows] == [1, 2, 3, 4, 5, 6]
