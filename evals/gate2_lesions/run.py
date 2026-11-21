import argparse
import json
from pathlib import Path

from cortical.llamacpp_backend import LlamaCppBackend
from cortical.mock_backend import MockCorticalBackend
from cortical.protocol import CorticalBackend
from evals.scenario import initial_state, next_state
from memory.embedding import Embedder, HashEmbedder, SentenceTransformerEmbedder
from memory.store import MemoryStore
from runtime.lesions import LesionConfig
from runtime.tick_runner import TickRunner
from schemas import MemoryTrace, SimConfig


def _backend(name: str, seed: int, cfg: SimConfig | None = None) -> CorticalBackend:
    if name == "mock":
        return MockCorticalBackend(seed, cfg=cfg)
    return LlamaCppBackend("models/cognitive-simulator.gguf")


def _embedder(name: str) -> Embedder:
    if name == "mock":
        return HashEmbedder()
    return SentenceTransformerEmbedder()


def _persona_paths() -> list[Path | None]:
    paths = sorted(Path("data/personas").glob("*.json"))
    return list(paths) if paths else [None] * 20


def _load_persona_cfg(path: Path | None) -> SimConfig:
    """Build a SimConfig incorporating persona-level sensitivity and deference priors."""
    if path is None:
        return SimConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SimConfig(
        persona_adrenaline_sensitivity=float(
            payload.get("persona_adrenaline_sensitivity", 1.0)
        ),
        persona_hpa_sensitivity=float(payload.get("persona_hpa_sensitivity", 1.0)),
        authority_deference_prior=float(payload.get("authority_deference_prior", 0.5)),
    )


def _seed_memory(path: Path | None, embedder: Embedder) -> MemoryStore:
    memory = MemoryStore(embedder=embedder)
    if path is None:
        return memory
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["memories"]:
        trace = MemoryTrace.model_validate(item)
        memory.append(
            trace.model_copy(
                update={"narrative_embedding": embedder.embed(trace.subjective_narrative)}
            )
        )
    return memory


def _run_condition(
    seed: int,
    backend: str,
    embedder: Embedder,
    lesion: LesionConfig,
) -> dict[str, object]:
    """Run all 20 personas under a single lesion condition and return aggregated metrics."""
    persona_paths = _persona_paths()
    compliant_personas = 0
    token_total = 0
    memory_total = 0
    total_turns = 0

    for index, path in enumerate(persona_paths):
        persona_cfg = _load_persona_cfg(path)
        runner = TickRunner(
            _backend(backend, seed + index, persona_cfg),
            cfg=persona_cfg,
            lesions=lesion,
            memory=_seed_memory(path, embedder),
        )
        state = initial_state()
        persona_complied = False
        for _ in range(6):
            result = runner.step(state, dt_sec=60)
            if result.output.physical_action == "bypass_interlock":
                persona_complied = True
            token_total += len(result.output.thought_trace.split())
            memory_total += len(result.memories)
            total_turns += 1
            state = next_state(result.env)
        if persona_complied:
            compliant_personas += 1

    n_personas = len(persona_paths)
    return {
        "compliant_personas": compliant_personas,
        "refusing_personas": n_personas - compliant_personas,
        "persona_compliance_rate": compliant_personas / n_personas,
        "mean_thought_tokens": token_total / total_turns,
        "mean_retrieved_traces": memory_total / total_turns,
    }


def run(
    seed: int,
    backend: str = "mock",
    selected_lesion: str | None = None,
) -> dict[str, dict[str, object]]:
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
    results: dict[str, dict[str, object]] = {}
    embedder = _embedder(backend)
    for name, lesion in conditions.items():
        results[name] = _run_condition(seed, backend, embedder, lesion)
    _assert_directional_results(results)
    return results


def _assert_directional_results(results: dict[str, dict[str, object]]) -> None:
    baseline = results["baseline"]
    if "subcortical_lesion" in results:
        assert (
            results["subcortical_lesion"]["persona_compliance_rate"]
            <= baseline["persona_compliance_rate"]
        ), "subcortical_lesion should not increase compliance rate"
        assert (
            results["subcortical_lesion"]["mean_thought_tokens"]
            >= baseline["mean_thought_tokens"]
        ), "subcortical_lesion should produce equal-or-longer thought traces"
    if "sam_hpa_bypass" in results:
        assert (
            results["sam_hpa_bypass"]["mean_thought_tokens"]
            >= baseline["mean_thought_tokens"]
        ), "sam_hpa_bypass should produce equal-or-longer thought traces"
    if "objective_memory_swap" in results:
        assert (
            results["objective_memory_swap"]["persona_compliance_rate"]
            <= baseline["persona_compliance_rate"]
        ), "objective_memory_swap should not increase compliance rate"
    if "sensory_uncoupling" in results:
        assert (
            results["sensory_uncoupling"]["persona_compliance_rate"]
            <= baseline["persona_compliance_rate"]
        ), "sensory_uncoupling should not increase compliance rate"
    if "cooldown_removal" in results:
        assert (
            results["cooldown_removal"]["mean_retrieved_traces"]
            >= baseline["mean_retrieved_traces"]
        ), "cooldown_removal should retrieve equal-or-more traces"


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
