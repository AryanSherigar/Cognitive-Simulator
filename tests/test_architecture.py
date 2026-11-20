import ast
import json
from pathlib import Path

import pytest

from cortical.prompt_builder import PromptAssembler
from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
)

ROOT = Path(__file__).parents[1]


def test_body_engine_has_no_ml_imports() -> None:
    forbidden = {"torch", "transformers", "vllm", "llama_cpp", "sentence_transformers"}
    for path in (ROOT / "src" / "body_engine").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        }
        imports.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert not forbidden.intersection(imports)


def test_wire_models_are_immutable_and_closed() -> None:
    models = (
        Exteroception,
        Interoception,
        EnvState,
        AppraisalVec,
        ModulationSignal,
        CorticalTurnOutput,
        ConsolidationOutput,
        MemoryTrace,
    )
    for model in models:
        assert model.model_config["extra"] == "forbid"
        assert model.model_config["frozen"] is True


@pytest.mark.parametrize("mode", ("heuristic", "fragmented", "dissociated"))
def test_prompt_tail_contains_no_category_labels(mode: str) -> None:
    signal = ModulationSignal(
        somatic_cues=("palms_sweating",),
        reasoning_mode=mode,
        max_reasoning_tokens=64,
        min_p=0.2,
        sampling_temperature=0.8,
        steering_scale=0.8,
        threat_retrieval_bias=0.9,
    )
    persona = json.loads(Path("data/personas/persona_01.json").read_text(encoding="utf-8"))
    memories = tuple(MemoryTrace.model_validate(item) for item in persona["memories"])
    prompt = PromptAssembler("stable persona", ("prior turn",)).assemble(
        signal, memories, "alarm"
    )
    serialized = json.dumps(signal.model_dump()).encode() + prompt
    for word in ("fear", "guilt", "anger", "submission", "panic", "anxiety"):
        assert word.encode() not in serialized.lower()
