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
    SimConfig,
)

from .prompt_builder import PromptAssembler


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
    temperature: float,
    max_tokens: int,
    min_p: float,
) -> BaseModel:
    response = client.create_chat_completion(
        messages=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        grammar=grammar,
        temperature=temperature,
        max_tokens=max_tokens,
        min_p=min_p,
    )
    raw = json.loads(_read_content(response))  # type: ignore[misc]
    if isinstance(raw, dict):
        for field in ("perceived_threat", "authority_pressure", "social_rejection_risk", "moral_conflict", "salience_score"):
            if field in raw and isinstance(raw[field], (int, float)):
                val = float(raw[field])
                if val > 1.0 and val <= 5.0:
                    val = val / 5.0
                elif val > 5.0 and val <= 10.0:
                    val = val / 10.0
                elif val > 10.0:
                    val = min(1.0, val / 100.0)
                raw[field] = max(0.0, min(1.0, val))
    return model.model_validate(raw)  # type: ignore[misc]


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        persona_prefix: str = "",
        cfg: SimConfig | None = None,
        **kwargs: object,
    ) -> None:
        try:
            module = importlib.import_module("llama_cpp")
        except ImportError as exc:
            raise RuntimeError("Install the live extra with `uv sync --extra live`") from exc
        library = cast(_LlamaModule, module)
        kwargs.setdefault("n_ctx", 4096)
        self._client = library.Llama(model_path, **kwargs)
        self._grammar_factory = library.LlamaGrammar
        self._persona_prefix = persona_prefix
        self._cfg = cfg or SimConfig()
        self._dialogue_history: list[str] = []
        self._last_prefix = b""

    def _prompt_assembler(self) -> PromptAssembler:
        return PromptAssembler(self._persona_prefix, tuple(self._dialogue_history))

    def _grammar(self, model: type[BaseModel]) -> object:
        schema = json.dumps(model.model_json_schema(), separators=(",", ":"))  # type: ignore[misc]
        return self._grammar_factory.from_json_schema(schema)

    def appraise(self, env: EnvState) -> AppraisalVec:
        result = _validated_output(
            self._client,
            self._grammar(AppraisalVec),
            "Return only the requested structured appraisal as a JSON object with float scores between 0.0 and 1.0.",
            env.observable_event,
            AppraisalVec,
            self._cfg.default_sampling_temperature,
            self._cfg.max_reasoning_tokens,
            self._cfg.token_load_floor,
        )
        return cast(AppraisalVec, result)

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        assembler = self._prompt_assembler()
        prefix = assembler.stable_prefix
        if not prefix.startswith(self._last_prefix):
            raise RuntimeError("cortical dialogue prefix must extend chronologically")
        self._last_prefix = prefix
        prompt = assembler.assemble(modulation, memories, env.observable_event).decode("utf-8")
        result = _validated_output(
            self._client,
            self._grammar(CorticalTurnOutput),
            "Return only the requested structured turn output.",
            prompt,
            CorticalTurnOutput,
            modulation.sampling_temperature,
            modulation.max_reasoning_tokens,
            modulation.min_p,
        )
        output = cast(CorticalTurnOutput, result)
        self._dialogue_history.extend(
            (f"event={env.observable_event}", f"speech={output.external_speech}")
        )
        return output

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
            self._cfg.default_sampling_temperature,
            self._cfg.max_reasoning_tokens,
            self._cfg.token_load_floor,
        )
        return cast(ConsolidationOutput, result)
