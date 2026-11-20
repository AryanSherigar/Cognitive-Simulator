import math

from schemas import Interoception, MemoryTrace, SimConfig


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def _somatic_similarity(left: Interoception, right: Interoception) -> float:
    values_left = (left.adrenaline, left.hpa_precursor, left.cortisol, left.fatigue, left.pain)
    values_right = (
        right.adrenaline,
        right.hpa_precursor,
        right.cortisol,
        right.fatigue,
        right.pain,
    )
    distance = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(values_left, values_right, strict=True))
    ) / math.sqrt(5)
    return 1.0 - min(1.0, distance)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norm if norm else 0.0


def score_trace(
    trace: MemoryTrace,
    sensory_tags: tuple[str, ...],
    somatic_state: Interoception,
    narrative_embedding: tuple[float, ...],
    threat_retrieval_bias: float,
    cfg: SimConfig,
) -> float:
    return (
        cfg.tag_score_weight * _jaccard(trace.sensory_tags, sensory_tags)
        + cfg.somatic_score_weight * _somatic_similarity(trace.somatic_signature, somatic_state)
        + cfg.embedding_score_weight * _cosine(trace.narrative_embedding, narrative_embedding)
        + cfg.threat_score_weight * trace.salience_score * threat_retrieval_bias
    )
