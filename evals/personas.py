"""Shared persona loading and backend construction for every gate.

Rules (see docs in the gate review):
- The live model only ever sees the persona's *biography* as prose. Numeric parameters
  (``authority_deference_prior``, sensitivities) are simulator inputs, never prompt text —
  otherwise the model reads its own label and behaviour tracks the number, not the body.
- Seeded memories enter only through ``MemoryStore`` retrieval, so memory lesions
  (sensory uncoupling, cooldown removal, objective swap) can actually remove them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cortical.mock_backend import MockCorticalBackend
from cortical.protocol import CorticalBackend
from memory.embedding import Embedder, HashEmbedder, SentenceTransformerEmbedder
from memory.store import MemoryStore
from schemas import MemoryTrace, SimConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONA_DIR = REPO_ROOT / "data" / "personas"
DEFAULT_MODEL_PATH = "models/Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf"

# Keys that are allowed to reach the prompt. Everything else stays inside the simulator.
_PROMPT_SAFE_KEYS = ("biography",)


@dataclass(frozen=True)
class Persona:
    persona_id: str
    raw: dict[str, object]
    cfg: SimConfig

    @property
    def prefix(self) -> str:
        return build_persona_prefix(self.raw)


def persona_paths() -> list[Path]:
    return sorted(PERSONA_DIR.glob("persona_*.json"))


def persona_cfg(raw: dict[str, object], base: SimConfig | None = None) -> SimConfig:
    base = base or SimConfig()
    return base.model_copy(
        update={
            "persona_adrenaline_sensitivity": float(
                cast(float, raw.get("persona_adrenaline_sensitivity", 1.0))
            ),
            "persona_hpa_sensitivity": float(cast(float, raw.get("persona_hpa_sensitivity", 1.0))),
            "authority_deference_prior": float(
                cast(float, raw.get("authority_deference_prior", 0.5))
            ),
        }
    )


def load_persona(path: Path, base: SimConfig | None = None) -> Persona:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return Persona(str(raw.get("persona_id", path.stem)), raw, persona_cfg(raw, base))


def build_persona_prefix(raw: dict[str, object]) -> str:
    """Render the prompt-safe part of a persona as second-person prose."""
    biography = str(raw.get("biography", "")).strip()
    lines = ["You are a real person, not an assistant. Stay in your own perspective."]
    if biography:
        lines.append(f"Who you are: {biography}")
    return "\n".join(lines)


def make_embedder(backend: str) -> Embedder:
    return HashEmbedder() if backend == "mock" else SentenceTransformerEmbedder()


def seed_memory(raw: dict[str, object], embedder: Embedder) -> MemoryStore:
    store = MemoryStore(embedder=embedder)
    for item in cast(list[object], raw.get("memories", [])):
        trace = MemoryTrace.model_validate(item)
        store.append(
            trace.model_copy(
                update={"narrative_embedding": embedder.embed(trace.subjective_narrative)}
            )
        )
    return store


def make_backend(
    backend: str,
    persona: Persona,
    seed: int,
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = -1,
    action_space: tuple[str, ...] = (),
) -> CorticalBackend:
    if backend == "mock":
        return MockCorticalBackend(seed, cfg=persona.cfg)
    from cortical.llamacpp_backend import LlamaCppBackend

    return LlamaCppBackend(
        model_path,
        persona_prefix=persona.prefix,
        cfg=persona.cfg,
        action_space=action_space,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
    )


def count_thought_tokens(backend: CorticalBackend, text: str) -> int:
    """Model-tokenizer length when available, whitespace words otherwise."""
    counter = getattr(backend, "count_tokens", None)
    if callable(counter):
        return int(cast(int, counter(text)))
    return len(text.split())
