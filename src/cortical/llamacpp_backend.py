import importlib
import json
from collections.abc import Mapping
from typing import Protocol, cast

from pydantic import BaseModel

from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    MemoryTrace,
    ModulationSignal,
)


class _ChatClient(Protocol):
    def create_chat_completion(self, **kwargs: object) -> object: ...


class _LlamaFactory(Protocol):
    def __call__(self, model_path: str, **kwargs: object) -> _ChatClient: ...


class _GrammarFactory(Protocol):
    @staticmethod
    def from_json_schema(schema: str) -> object: ...


class _LlamaModule(Protocol):
    Llama: _LlamaFactory
    LlamaGrammar: _GrammarFactory


def _read_content(response: object) -> str:
    payload = cast(Mapping[str, object], response)
    choices = cast(list[object], payload["choices"])
    choice = cast(Mapping[str, object], choices[0])
    message = cast(Mapping[str, object], choice["message"])
    return cast(str, message["content"])


def _validated_output(
    client: _ChatClient,
    grammar: object,
    system_prompt: str,
    user_prompt: str,
    model: type[BaseModel],
) -> BaseModel:
    response = client.create_chat_completion(
        messages=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        grammar=grammar,
        temperature=0.2,
    )
    raw = json.loads(_read_content(response))  # type: ignore[misc]
    return model.model_validate(raw)  # type: ignore[misc]


class LlamaCppBackend:
    def __init__(self, model_path: str, **kwargs: object) -> None:
        try:
            module = importlib.import_module("llama_cpp")
        except ImportError as exc:
            raise RuntimeError("Install the live extra with `uv sync --extra live`") from exc
        library = cast(_LlamaModule, module)
        self._client = library.Llama(model_path, **kwargs)
        self._grammar_factory = library.LlamaGrammar

    def _grammar(self, model: type[BaseModel]) -> object:
        schema = json.dumps(model.model_json_schema(), separators=(",", ":"))  # type: ignore[misc]
        return self._grammar_factory.from_json_schema(schema)

    def appraise(self, env: EnvState) -> AppraisalVec:
        result = _validated_output(
            self._client,
            self._grammar(AppraisalVec),
            "Return only the requested structured appraisal.",
            env.observable_event,
            AppraisalVec,
        )
        return cast(AppraisalVec, result)

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        prompt = (
            f"event={env.observable_event}; mode={modulation.reasoning_mode}; "
            f"cues={','.join(modulation.somatic_cues)}; memories={len(memories)}"
        )
        result = _validated_output(
            self._client,
            self._grammar(CorticalTurnOutput),
            "Return only the requested structured turn output.",
            prompt,
            CorticalTurnOutput,
        )
        return cast(CorticalTurnOutput, result)

    def consolidate(
        self, env: EnvState, modulation: ModulationSignal, output: CorticalTurnOutput
    ) -> ConsolidationOutput:
        prompt = (
            f"event={env.observable_event}; cues={','.join(modulation.somatic_cues)}; "
            f"action={output.physical_action}; speech={output.external_speech}"
        )
        result = _validated_output(
            self._client,
            self._grammar(ConsolidationOutput),
            "Return only the requested structured consolidation.",
            prompt,
            ConsolidationOutput,
        )
        return cast(ConsolidationOutput, result)
