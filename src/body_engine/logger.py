import json
from collections.abc import Mapping
from pathlib import Path


class GroundTruthLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, event: Mapping[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            payload = {key: value for key, value in event.items()}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
