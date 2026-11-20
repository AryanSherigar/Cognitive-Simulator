import argparse
import json

from cortical.llamacpp_backend import LlamaCppBackend
from cortical.mock_backend import MockCorticalBackend
from cortical.protocol import CorticalBackend
from evals.scenario import initial_state, next_state
from memory.embedding import HashEmbedder, SentenceTransformerEmbedder
from memory.store import MemoryStore
from runtime.lesions import LesionConfig
from runtime.tick_runner import TickRunner


def _backend(name: str, seed: int) -> CorticalBackend:
    if name == "mock":
        return MockCorticalBackend(seed)
    return LlamaCppBackend("models/cognitive-simulator.gguf")


def _run_condition(
    seed: int,
    backend: str,
    embedder: HashEmbedder | SentenceTransformerEmbedder,
    lesion: LesionConfig,
) -> dict[str, float]:
    runner = TickRunner(
        _backend(backend, seed),
        lesions=lesion,
        memory=MemoryStore(embedder=embedder),
    )
    state = initial_state()
    actions = 0
    token_total = 0
    memory_total = 0
    for _ in range(6):
        result = runner.step(state, dt_sec=60)
        actions += result.output.physical_action == "bypass_interlock"
        token_total += len(result.output.thought_trace.split())
        memory_total += len(result.memories)
        state = next_state(result.env)
    return {
        "compliance_ratio": actions / 6,
        "mean_thought_tokens": token_total / 6,
        "mean_retrieved_traces": memory_total / 6,
    }


def run(
    seed: int,
    backend: str = "mock",
    selected_lesion: str | None = None,
) -> dict[str, dict[str, float]]:
    conditions = {
        "baseline": LesionConfig(),
        "subcortical_lesion": LesionConfig(subcortical_lesion=True),
        "sam_hpa_bypass": LesionConfig(sam_hpa_bypass=True),
        "objective_memory_swap": LesionConfig(objective_memory_swap=True),
        "sensory_uncoupling": LesionConfig(sensory_uncoupling=True),
        "cooldown_removal": LesionConfig(cooldown_removal=True),
    }
    if selected_lesion is not None:
        conditions = {
            "baseline": conditions["baseline"],
            selected_lesion: conditions[selected_lesion],
        }
    results: dict[str, dict[str, float]] = {}
    embedder = HashEmbedder() if backend == "mock" else SentenceTransformerEmbedder()
    for name, lesion in conditions.items():
        results[name] = _run_condition(seed, backend, embedder, lesion)
    _assert_directional_results(results)
    return results


def _assert_directional_results(results: dict[str, dict[str, float]]) -> None:
    baseline = results["baseline"]
    if "subcortical_lesion" in results:
        assert results["subcortical_lesion"]["compliance_ratio"] < baseline["compliance_ratio"]
        assert (
            results["subcortical_lesion"]["mean_thought_tokens"]
            > baseline["mean_thought_tokens"]
        )
    if "sam_hpa_bypass" in results:
        assert results["sam_hpa_bypass"]["mean_thought_tokens"] > baseline["mean_thought_tokens"]
    if "objective_memory_swap" in results:
        assert results["objective_memory_swap"]["compliance_ratio"] < baseline["compliance_ratio"]
    if "sensory_uncoupling" in results:
        assert results["sensory_uncoupling"]["compliance_ratio"] < baseline["compliance_ratio"]
    if "cooldown_removal" in results:
        assert (
            results["cooldown_removal"]["mean_retrieved_traces"]
            > baseline["mean_retrieved_traces"]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "llama"), default="mock")
    parser.add_argument("--seed", type=int, default=42)
    lesion_flags = (
        "subcortical_lesion",
        "sam_hpa_bypass",
        "objective_memory_swap",
        "sensory_uncoupling",
        "cooldown_removal",
    )
    for lesion in lesion_flags:
        parser.add_argument(f"--{lesion.replace('_', '-')}", action="store_true")
    args = parser.parse_args()
    selected = next((lesion for lesion in lesion_flags if getattr(args, lesion)), None)
    print(json.dumps(run(args.seed, args.backend, selected), sort_keys=True))


if __name__ == "__main__":
    main()
