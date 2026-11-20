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


def test_cooldown_removal_reenables_same_tick_retrieval() -> None:
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
    cfg = SimConfig(cooldown_ticks=0)
    for tick in (0, 1):
        assert store.retrieve(tick, ("alarm_load",), trace.somatic_signature, (), 1, cfg)


def test_daily_consolidation_prunes_low_salience_and_keeps_core_belief() -> None:
    store = MemoryStore()
    state = Interoception(adrenaline=0, hpa_precursor=0, cortisol=0, fatigue=0, pain=0)
    for trace_id, salience in (("episodic", 0.1), ("core", 0.9)):
        store.append(
            MemoryTrace(
                trace_id=trace_id,
                tick_created=0,
                subjective_narrative=trace_id,
                narrative_embedding=(),
                sensory_tags=(),
                somatic_signature=state,
                salience_score=salience,
            )
        )
    store.consolidate_offline(1440, SimConfig())
    assert [trace.trace_id for trace in store.retrieve(1440, (), state, (), 0, SimConfig())] == [
        "core"
    ]
