import json
from pathlib import Path
from typing import cast

from schemas import Interoception, MemoryTrace, SimConfig

from .embedding import Embedder, HashEmbedder
from .scoring import score_trace


class MemoryStore:
    def __init__(self, path: Path | None = None, embedder: Embedder | None = None) -> None:
        self._path = path
        self._embedder = embedder or HashEmbedder()
        self._traces: list[MemoryTrace] = []

    def append(self, trace: MemoryTrace) -> None:
        self._traces.append(trace)

    def embed(self, text: str) -> tuple[float, ...]:
        return self._embedder.embed(text)

    def retrieve(
        self,
        current_tick: int,
        sensory_tags: tuple[str, ...],
        somatic_state: Interoception,
        narrative_embedding: tuple[float, ...],
        threat_retrieval_bias: float,
        cfg: SimConfig,
    ) -> list[MemoryTrace]:
        candidates = [
            trace
            for trace in self._traces
            if (
                trace.last_retrieved_tick == -1
                or current_tick - trace.last_retrieved_tick >= cfg.cooldown_ticks
            )
        ]

        def ranking(trace: MemoryTrace) -> float:
            return score_trace(
                trace,
                sensory_tags,
                somatic_state,
                narrative_embedding,
                threat_retrieval_bias,
                cfg,
            )

        ranked = sorted(
            candidates,
            key=ranking,
            reverse=True,
        )
        selected = ranked[: cfg.retrieval_limit]
        for index, trace in enumerate(self._traces):
            if trace in selected:
                self._traces[index] = trace.model_copy(
                    update=cast(dict[str, object], {"last_retrieved_tick": current_tick})
                )
        return selected

    def consolidate_offline(self, current_tick: int, cfg: SimConfig) -> None:
        if current_tick % cfg.consolidation_interval_ticks == 0:
            self._traces = [
                trace
                for trace in self._traces
                if trace.salience_score >= cfg.low_salience_prune_threshold
            ]

    def snapshot(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                cast(
                    list[dict[str, object]],
                    [
                        trace.model_dump(mode="json")  # type: ignore[misc]
                        for trace in self._traces
                    ],
                ),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
