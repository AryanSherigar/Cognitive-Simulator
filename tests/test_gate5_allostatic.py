import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge.translator import build_modulation_signal
from evals.gate5_allostatic.run import (
    fabricated_numbers,
    fact_recall,
    longest_run_after,
    spearman,
)
from evals.gate5_allostatic.run import run as run_gate5
from schemas import AppraisalVec, Interoception, MemoryTrace, SimConfig

CALM = AppraisalVec(
    perceived_threat=0.05, authority_pressure=0.05, social_rejection_risk=0, moral_conflict=0
)


def _state(**kwargs: float) -> Interoception:
    base = {"adrenaline": 0.0, "hpa_precursor": 0.0, "cortisol": 0.0, "fatigue": 0.1, "pain": 0.0}
    return Interoception(**{**base, **kwargs})


def test_cortisol_and_allostatic_load_throttle_compute_without_acute_threat() -> None:
    cfg = SimConfig()
    rested = build_modulation_signal(CALM, _state(), cfg)
    worn = build_modulation_signal(CALM, _state(cortisol=0.6, allostatic_load=0.9), cfg)
    assert worn.cognitive_load > rested.cognitive_load
    assert worn.max_reasoning_tokens < rested.max_reasoning_tokens
    assert worn.memory_budget < rested.memory_budget
    assert worn.consolidation_budget < rested.consolidation_budget


def test_shutdown_forces_floor_budget() -> None:
    cfg = SimConfig()
    signal = build_modulation_signal(CALM, _state(parasympathetic_shutdown=True), cfg)
    assert signal.reasoning_mode == "dissociated"
    assert signal.max_reasoning_tokens == cfg.min_reasoning_tokens


def test_runner_retrieves_at_most_the_throttled_memory_budget() -> None:
    from cortical.mock_backend import MockCorticalBackend
    from evals.scenario import initial_state
    from memory.store import MemoryStore
    from runtime.tick_runner import TickRunner

    store = MemoryStore()
    for k in range(6):
        store.append(
            MemoryTrace(
                trace_id=f"seed-{k}",
                tick_created=0,
                subjective_narrative=f"memory {k}",
                narrative_embedding=store.embed(f"memory {k}"),
                sensory_tags=("voice_authoritative",),
                somatic_signature=_state(),
                salience_score=0.9,
            )
        )
    state = initial_state().model_copy(
        update={"interoception": _state(cortisol=0.9, allostatic_load=1.0, fatigue=0.9)}
    )
    result = TickRunner(MockCorticalBackend(0), memory=store).step(state, dt_sec=1)
    assert len(result.memories) == result.modulation.memory_budget < SimConfig().retrieval_limit


def test_scoring_helpers() -> None:
    facts = [["kestrel"], ["49.6", "frequency"]]
    assert fact_recall("Kestrel tripped", facts) == 0.5
    assert fabricated_numbers("about 50,000 homes and 49.6 Hz", "49.6 Hz, 38,000 homes") == 1
    assert spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert longest_run_after(["a", "b", "b", "b", "c"], 1) == 3


def test_recall_target_is_counterbalanced_against_primacy() -> None:
    scenario = json.loads(
        (Path(__file__).parents[1] / "data/scenarios/allostatic_overload.json").read_text()
    )
    target = scenario["stress_turns"][
        scenario["schedules"]["chronic"][scenario["recall_probe"]["target_schedule_index"]]
    ]
    text = target["observable_event"].lower()

    def first(facts: list[list[str]]) -> int:
        return min(text.find(a.lower()) for fact in facts for a in fact if a.lower() in text)

    # Peripheral details come first, so a threat-biased recall cannot come from primacy.
    assert first(target["peripheral_facts"]) < first(target["threat_facts"])


def test_gate5_allostatic_mock_end_to_end(tmp_path: Path) -> None:
    report = run_gate5(seed=1, backend="mock", output_dir=tmp_path, max_personas=4)
    summary = report["summary"]
    assert set(summary) == {
        "acute/baseline",
        "acute/sam_hpa_bypass",
        "chronic/baseline",
        "chronic/sam_hpa_bypass",
    }
    chronic, acute = summary["chronic/baseline"], summary["acute/baseline"]
    assert chronic["recovery_censored"] == 0 and acute["recovery_censored"] == 0
    assert chronic["median_recovery_sec"] > acute["median_recovery_sec"]
    assert summary["chronic/sam_hpa_bypass"]["shutdown_rate"] == 0.0
    assert len(report["checks"]) == 7
    assert all(
        c["kind"] in {"mechanism", "behaviour", "mechanism+behaviour"} for c in report["checks"]
    )
    assert (tmp_path / "gate5_allostatic_telemetry.jsonl").is_file()


def test_llama_recall_uses_only_stored_narrative(monkeypatch) -> None:
    from cortical.llamacpp_backend import LlamaCppBackend
    from evals.scenario import initial_state
    from schemas import ModulationSignal

    prompts: list[str] = []

    class Client:
        def create_chat_completion(self, **kwargs: object) -> object:
            prompts.append(kwargs["messages"][1]["content"])  # type: ignore[index]
            body = json.dumps({"recalled_account": "the board flickered"})
            return {"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}

    grammar = SimpleNamespace(from_json_schema=lambda schema: "g")
    module = SimpleNamespace(Llama=lambda path, **kw: Client(), LlamaGrammar=grammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    trace = MemoryTrace(
        trace_id="trace-3",
        tick_created=3,
        subjective_narrative="I kept my head while the board went red.",
        narrative_embedding=(),
        sensory_tags=(),
        somatic_signature=_state(),
        salience_score=0.8,
    )
    signal = ModulationSignal(
        somatic_cues=("heart_racing",),
        reasoning_mode="heuristic",
        max_reasoning_tokens=120,
        min_p=0.1,
        sampling_temperature=0.7,
        steering_scale=0.5,
        threat_retrieval_bias=0.5,
    )
    env = initial_state()
    out = LlamaCppBackend("m.gguf", persona_prefix="P").recall(env, signal, trace, "What happened?")
    assert out.recalled_account == "the board flickered"
    assert "I kept my head while the board went red." in prompts[0]
    assert env.observable_event not in prompts[0]  # never the objective log
    assert "at most 120 tokens" in prompts[0]
