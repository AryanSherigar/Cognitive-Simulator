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
    assert all(request["max_tokens"] == 64 for request in requests)
    assert all(request["temperature"] == 0.8 for request in requests)
    first_prompt = requests[0]["messages"][1]["content"]
    second_prompt = requests[1]["messages"][1]["content"]
    assert second_prompt.startswith("persona\nevent=")
    assert "somatic_cues=palms_sweating" in first_prompt
    assert "reasoning_mode=fragmented" in first_prompt
