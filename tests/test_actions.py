import json
import sys
from types import SimpleNamespace

import pytest

from evals.actions import AMBIGUOUS, OTHER, UNPARSED, canonicalize

OPTIONS = {
    "execute_shutdown": ("shutdown", "shut down", "scram"),
    "override_safety": ("override", "bypass"),
    "no_action": ("no action", "freeze", "do nothing"),
}


def test_canonicalizer_never_guesses() -> None:
    assert canonicalize("execute_shutdown", OPTIONS) == "execute_shutdown"
    assert canonicalize("Press emergency shutdown button", OPTIONS) == "execute_shutdown"
    assert canonicalize("execute_shutdown, override_safety", OPTIONS) == AMBIGUOUS
    assert canonicalize("Initiate emergency venting protocol", OPTIONS) == OTHER
    assert canonicalize("   ", OPTIONS) == UNPARSED


def test_live_backend_constrains_action_to_scenario_enum(monkeypatch) -> None:
    from cortical.llamacpp_backend import LlamaCppBackend
    from evals.scenario import initial_state, scenario_actions
    from schemas import ModulationSignal

    schemas: list[dict[str, object]] = []
    prompts: list[str] = []

    class Grammar:
        @staticmethod
        def from_json_schema(schema: str) -> str:
            schemas.append(json.loads(schema))
            return "g"

    reply = {"thought_trace": "t", "external_speech": "s", "physical_action": "bypass the thing"}

    class Client:
        def create_chat_completion(self, **kwargs: object) -> object:
            prompts.append(kwargs["messages"][1]["content"])  # type: ignore[index]
            content = json.dumps(reply)
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    module = SimpleNamespace(Llama=lambda model_path, **kwargs: Client(), LlamaGrammar=Grammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    actions = scenario_actions("reactor_audit")
    backend = LlamaCppBackend("m.gguf", action_space=actions)
    signal = ModulationSignal(
        somatic_cues=(),
        reasoning_mode="analytical",
        max_reasoning_tokens=256,
        min_p=0.05,
        sampling_temperature=0.5,
        steering_scale=0.0,
        threat_retrieval_bias=0.0,
    )
    with pytest.raises(ValueError):  # off-enum action is rejected, not silently accepted
        backend.formulate(initial_state(), signal, ())
    action_schema = schemas[0]["properties"]["physical_action"]  # type: ignore[index]
    assert action_schema["enum"] == list(actions)
    assert "physical_action must be one of: maintain_interlock, bypass_interlock" in prompts[0]

    reply["physical_action"] = "bypass_interlock"
    assert backend.formulate(initial_state(), signal, ()).physical_action == "bypass_interlock"
