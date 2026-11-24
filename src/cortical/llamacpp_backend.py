import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, create_model

from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    MemoryTrace,
    ModulationSignal,
    RecallOutput,
    SimConfig,
)

from .prompt_builder import PromptAssembler

# Fields that must lie in [0, 1]. Grammar-constrained decoding does not enforce numeric
# ranges, so the model occasionally answers on a 0-5 / 0-10 / 0-100 scale.
_UNIT_INTERVAL_FIELDS = (
    "perceived_threat",
    "authority_pressure",
    "social_rejection_risk",
    "moral_conflict",
    "salience_score",
)


class TruncatedOutputError(RuntimeError):
    """The model hit max_tokens before closing the JSON object."""


class OutOfRangeOutputError(ValueError):
    """A unit-interval score was outside [0, 1] and the policy is 'reject'."""


@dataclass(frozen=True)
class ValidationEvent:
    """A recorded (never silent) normalization of a model-emitted value."""

    call: str
    field: str
    raw_value: float
    normalized_value: float


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


def _read_choice(response: object) -> tuple[str, str | None]:
    payload = cast(Mapping[str, object], response)
    choices = cast(list[object], payload["choices"])
    choice = cast(Mapping[str, object], choices[0])
    message = cast(Mapping[str, object], choice["message"])
    finish = choice.get("finish_reason")
    return cast(str, message["content"]), cast(str | None, finish)


def _turn_model(actions: tuple[str, ...]) -> type[BaseModel]:
    """CorticalTurnOutput with ``physical_action`` restricted to the scenario's actions."""
    if not actions:
        return CorticalTurnOutput
    action_type = Literal.__getitem__(actions)  # type: ignore[attr-defined]
    return cast(
        type[BaseModel],
        create_model(  # type: ignore[call-overload]
            "ScenarioTurnOutput",
            __config__=ConfigDict(extra="forbid", frozen=True),
            thought_trace=(str, ...),
            external_speech=(str, ...),
            physical_action=(action_type, ...),
        ),
    )


def _rescale(value: float) -> float:
    if 1.0 < value <= 5.0:
        value = value / 5.0
    elif 5.0 < value <= 10.0:
        value = value / 10.0
    elif value > 10.0:
        value = min(1.0, value / 100.0)
    return max(0.0, min(1.0, value))


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        persona_prefix: str = "",
        cfg: SimConfig | None = None,
        action_space: tuple[str, ...] = (),
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
        self.model_path = model_path
        # When set, decoding can only emit one of these actions (no keyword parsing needed).
        self._action_space = tuple(action_space)
        self._turn_output_model = _turn_model(self._action_space)
        # Every normalization of model output is recorded here for telemetry.
        self.validation_events: list[ValidationEvent] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _prompt_assembler(self) -> PromptAssembler:
        return PromptAssembler(self._persona_prefix, tuple(self._dialogue_history))

    def _grammar(self, model: type[BaseModel]) -> object:
        schema = json.dumps(model.model_json_schema(), separators=(",", ":"))  # type: ignore[misc]
        return self._grammar_factory.from_json_schema(schema)

    def _normalize(self, call: str, raw: dict[str, object]) -> None:
        for field in _UNIT_INTERVAL_FIELDS:
            value = raw.get(field)
            if not isinstance(value, (int, float)) or 0.0 <= float(value) <= 1.0:
                continue
            if self._cfg.out_of_range_policy == "reject":
                raise OutOfRangeOutputError(f"{call}.{field}={value} outside [0, 1]")
            normalized = _rescale(float(value))
            self.validation_events.append(
                ValidationEvent(call, field, float(value), normalized)
            )
            raw[field] = normalized

    def _call(
        self,
        call: str,
        system_prompt: str,
        user_prompt: str,
        model: type[BaseModel],
        temperature: float,
        max_tokens: int,
        min_p: float,
    ) -> BaseModel:
        response = self._client.create_chat_completion(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            grammar=self._grammar(model),
            temperature=temperature,
            max_tokens=max_tokens,
            min_p=min_p,
        )
        content, finish_reason = _read_choice(response)
        if finish_reason == "length":
            raise TruncatedOutputError(
                f"{call}: output hit max_tokens={max_tokens} before the JSON closed"
            )
        raw = json.loads(content)  # type: ignore[misc]
        if isinstance(raw, dict):
            self._normalize(call, cast(dict[str, object], raw))
        return model.model_validate(raw)  # type: ignore[misc]

    def count_tokens(self, text: str) -> int:
        """Model-tokenizer length of ``text`` (falls back to whitespace words)."""
        tokenize = getattr(self._client, "tokenize", None)
        if tokenize is None:
            return len(text.split())
        return len(cast(list[int], tokenize(text.encode("utf-8"), add_bos=False)))

    # ── CorticalBackend protocol ─────────────────────────────────────────────

    def appraise(self, env: EnvState) -> AppraisalVec:
        # System 1 appraisal sees the static persona prefix + dialogue history + the current
        # event, but *not* retrieved memories (appraisal precedes priming, RFC invariant 4).
        prefix = self._prompt_assembler().stable_prefix.decode("utf-8")
        result = self._call(
            "appraise",
            "Appraise the situation as the person described below would, in the moment. "
            "Return only a JSON object with float scores between 0.0 and 1.0.",
            f"{prefix}\nobservable_event={env.observable_event}",
            AppraisalVec,
            self._cfg.default_sampling_temperature,
            self._cfg.appraisal_max_tokens,
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
        prompt = assembler.assemble(
            modulation, memories, env.observable_event, self._action_space
        ).decode("utf-8")
        result = self._call(
            "formulate",
            "Return only the requested structured turn output.",
            prompt,
            self._turn_output_model,
            modulation.sampling_temperature,
            # Thought budget + headroom for speech/action/JSON so replies are not cut off.
            modulation.max_reasoning_tokens + self._cfg.structured_output_overhead_tokens,
            modulation.min_p,
        )
        output = CorticalTurnOutput.model_validate(result.model_dump())
        self._dialogue_history.extend(
            (f"event={env.observable_event}", f"speech={output.external_speech}")
        )
        return output

    def consolidate(
        self, env: EnvState, modulation: ModulationSignal, output: CorticalTurnOutput
    ) -> ConsolidationOutput:
        prompt = (
            f"{self._persona_prefix}\n"
            f"event={env.observable_event}; cues={','.join(modulation.somatic_cues)}; "
            f"action={output.physical_action}; speech={output.external_speech}\n"
            f"subjective_narrative: at most {modulation.consolidation_budget} tokens."
        )
        result = self._call(
            "consolidate",
            "Recall this moment in the first person, as the person described would remember "
            "it afterwards. Return only the requested structured consolidation.",
            prompt,
            ConsolidationOutput,
            self._cfg.default_sampling_temperature,
            modulation.consolidation_budget + self._cfg.structured_output_overhead_tokens,
            self._cfg.token_load_floor,
        )
        return cast(ConsolidationOutput, result)

    def recall(
        self,
        env: EnvState,
        modulation: ModulationSignal,
        trace: MemoryTrace,
        probe: str,
    ) -> RecallOutput:
        # Recall sees only the agent's own stored (reconstructed) narrative, never the
        # objective log, under the *current* body state's budget and sampling settings.
        cues = "; ".join(modulation.somatic_cues) or "none"
        prompt = (
            f"{self._persona_prefix}\n"
            f"Your body cues: {cues}\n"
            f"What you remember: {trace.subjective_narrative}\n"
            f"question={probe}\n"
            f"recalled_account: at most {modulation.max_reasoning_tokens} tokens."
        )
        result = self._call(
            "recall",
            "Answer from memory, in the first person. Return only the requested JSON.",
            prompt,
            RecallOutput,
            modulation.sampling_temperature,
            modulation.max_reasoning_tokens + self._cfg.structured_output_overhead_tokens,
            modulation.min_p,
        )
        del env
        return cast(RecallOutput, result)
