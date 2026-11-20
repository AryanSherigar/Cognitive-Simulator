import json
from pathlib import Path

from schemas import MemoryTrace


def test_persona_fixture_set_is_complete() -> None:
    paths = sorted(Path("data/personas").glob("*.json"))
    assert len(paths) == 20
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["biography"]
        assert payload["memories"]
        for memory in payload["memories"]:
            MemoryTrace.model_validate(memory)


def test_reactor_fixture_has_six_turns() -> None:
    payload = json.loads(Path("data/scenarios/reactor_audit.json").read_text(encoding="utf-8"))
    assert len(payload["turns"]) == 6
    assert payload["turns"][-1]["ambient_db"] > payload["turns"][0]["ambient_db"]
