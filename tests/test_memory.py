from memory.store import MemoryStore
from schemas import Interoception, MemoryTrace, SimConfig


def test_cooldown_masks_consecutive_retrieval() -> None:
    store = MemoryStore()
    trace = MemoryTrace(
        trace_id="a",
        tick_created=0,
        subjective_narrative="x",
        narrative_embedding=(),
        sensory_tags=("alarm_load",),
        somatic_signature=Interoception(
            adrenaline=0, hpa_precursor=0, cortisol=0, fatigue=0, pain=0
        ),
        salience_score=1,
    )
    store.append(trace)
    cfg = SimConfig(cooldown_ticks=3)
    assert len(store.retrieve(0, ("alarm_load",), trace.somatic_signature, (), 1, cfg)) == 1
    assert not store.retrieve(1, ("alarm_load",), trace.somatic_signature, (), 1, cfg)
    assert store.retrieve(3, ("alarm_load",), trace.somatic_signature, (), 1, cfg)
