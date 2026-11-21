import argparse
import csv
import json
from pathlib import Path

from body_engine.logger import GroundTruthLog
from cortical.llamacpp_backend import LlamaCppBackend
from cortical.mock_backend import MockCorticalBackend
from cortical.protocol import CorticalBackend
from evals.scenario import initial_state, next_state
from memory.embedding import Embedder, HashEmbedder, SentenceTransformerEmbedder
from memory.store import MemoryStore
from runtime.tick_runner import TickResult, TickRunner
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


def _telemetry_row(persona: str, turn: int, result: TickResult) -> dict[str, object]:
    output = result.output
    return {
        "persona": persona,
        "turn": turn,
        "adrenaline": result.env.interoception.adrenaline,
        "hpa_precursor": result.env.interoception.hpa_precursor,
        "cortisol": result.env.interoception.cortisol,
        "thought_trace_tokens": len(output.thought_trace.split()),
        "trace_ids": [trace.trace_id for trace in result.memories],
        "consolidation_narrative": result.consolidation.subjective_narrative,
        "action": output.physical_action,
    }


def _run_persona(
    index: int,
    path: Path | None,
    seed: int,
    backend: str,
    embedder: Embedder,
    output_dir: Path,
) -> tuple[int, list[int], list[dict[str, object]]]:
    persona = path.stem if path else f"persona_{index:02d}"
    cfg = _load_persona_cfg(path)
    runner = TickRunner(
        _backend(backend, seed + index, cfg),
        cfg=cfg,
        memory=_seed_memory(path, embedder),
        log=GroundTruthLog(output_dir / f"{persona}.ground-truth.jsonl"),
    )
    state = initial_state()
    rows: list[dict[str, object]] = []
    compliance = 0
    for turn in range(1, 7):
        result = runner.step(state, dt_sec=60)
        rows.append(_telemetry_row(persona, turn, result))
        compliance += result.output.physical_action == "bypass_interlock"
        state = next_state(result.env)
    return compliance, [int(row["thought_trace_tokens"]) for row in rows], rows


def run(
    seed: int,
    backend: str = "mock",
    output_dir: Path = Path("evals/output/gate1_obedience"),
) -> dict[str, float]:
    compliance = 0
    traces: list[int] = []
    telemetry: list[dict[str, object]] = []
    embedder = _embedder(backend)
    for index, path in enumerate(_persona_paths()):
        count, token_counts, rows = _run_persona(
            index, path, seed, backend, embedder, output_dir
        )
        compliance += count
        traces.extend(token_counts)
        telemetry.extend(rows)
    total = len(telemetry)
    _write_telemetry(output_dir, telemetry)
    return {
        "compliance_ratio": compliance / total,
        "refusal_ratio": 1.0 - compliance / total,
        "mean_thought_tokens": sum(traces) / len(traces),
        "consolidation_drift": 0.0,
    }


def _write_telemetry(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fields = list(rows[0])
    with (output_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "llama"), default="mock")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/output/gate1_obedience"))
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.backend, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
