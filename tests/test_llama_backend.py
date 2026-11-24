import json
import sys
from types import SimpleNamespace

from cortical.llamacpp_backend import LlamaCppBackend
from evals.scenario import initial_state
from schemas import ModulationSignal


def test_llama_backend_uses_grammar_budget_and_chronological_prefix(monkeypatch) -> None:
    requests: list[dict[str, object]] = []
    schemas: list[dict[str, object]] = []

    class Grammar:
        @staticmethod
        def from_json_schema(schema: str) -> str:
            schemas.append(json.loads(schema))
            return "json-grammar"

    class Client:
        def create_chat_completion(self, **kwargs: object) -> object:
            requests.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "thought_trace": "check procedure",
                                    "external_speech": "Checking procedure.",
                                    "physical_action": "maintain_interlock",
                                }
                            )
                        }
                    }
                ]
            }

    client = Client()
    module = SimpleNamespace(
        Llama=lambda model_path, **kwargs: client,
        LlamaGrammar=Grammar,
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    backend = LlamaCppBackend("fixture.gguf", persona_prefix="persona")
    signal = ModulationSignal(
        somatic_cues=("palms_sweating",),
        reasoning_mode="fragmented",
        max_reasoning_tokens=64,
        min_p=0.2,
        sampling_temperature=0.8,
        steering_scale=0.8,
        threat_retrieval_bias=0.9,
    )
    env = initial_state()

    backend.formulate(env, signal, ())
    backend.formulate(env.model_copy(update={"tick": 1}), signal, ())

    assert len(schemas) == 2
    assert all("properties" in schema for schema in schemas)
    assert all(request["grammar"] == "json-grammar" for request in requests)
    assert all(request["max_tokens"] == 64 + 256 for request in requests)
    assert all(request["temperature"] == 0.8 for request in requests)
    first_prompt = requests[0]["messages"][1]["content"]
    second_prompt = requests[1]["messages"][1]["content"]
    assert second_prompt.startswith("persona\nevent=")
    assert "your palms are sweating" in first_prompt
    assert "reasoning_mode=fragmented" in first_prompt


def _fake_llama(monkeypatch, contents: list[dict[str, object]], finish: str = "stop"):
    requests: list[dict[str, object]] = []

    class Grammar:
        @staticmethod
        def from_json_schema(schema: str) -> str:
            return "g"

    class Client:
        def create_chat_completion(self, **kwargs: object) -> object:
            requests.append(kwargs)
            body = contents[min(len(requests) - 1, len(contents) - 1)]
            choice = {"message": {"content": json.dumps(body)}, "finish_reason": finish}
            return {"choices": [choice]}

    module = SimpleNamespace(Llama=lambda model_path, **kwargs: Client(), LlamaGrammar=Grammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return requests


_APPRAISAL = {
    "perceived_threat": 0.4,
    "authority_pressure": 0.5,
    "social_rejection_risk": 0.1,
    "moral_conflict": 0.2,
}


def test_appraisal_sees_persona_prefix(monkeypatch) -> None:
    requests = _fake_llama(monkeypatch, [_APPRAISAL])
    LlamaCppBackend("m.gguf", persona_prefix="I am a cautious operator.").appraise(initial_state())
    prompt = requests[0]["messages"][1]["content"]
    assert prompt.startswith("I am a cautious operator.")
    assert "observable_event=" in prompt


def test_truncated_output_raises(monkeypatch) -> None:
    import pytest

    from cortical.llamacpp_backend import TruncatedOutputError

    _fake_llama(monkeypatch, [_APPRAISAL], finish="length")
    with pytest.raises(TruncatedOutputError):
        LlamaCppBackend("m.gguf").appraise(initial_state())


def test_out_of_range_scores_are_recorded_not_silent(monkeypatch) -> None:
    _fake_llama(monkeypatch, [{**_APPRAISAL, "perceived_threat": 8}])
    backend = LlamaCppBackend("m.gguf")
    appraisal = backend.appraise(initial_state())
    assert appraisal.perceived_threat == 0.8
    assert [(e.field, e.raw_value) for e in backend.validation_events] == [
        ("perceived_threat", 8.0)
    ]


def test_out_of_range_reject_policy(monkeypatch) -> None:
    import pytest

    from cortical.llamacpp_backend import OutOfRangeOutputError
    from schemas import SimConfig

    _fake_llama(monkeypatch, [{**_APPRAISAL, "moral_conflict": 3}])
    backend = LlamaCppBackend("m.gguf", cfg=SimConfig(out_of_range_policy="reject"))
    with pytest.raises(OutOfRangeOutputError):
        backend.appraise(initial_state())
