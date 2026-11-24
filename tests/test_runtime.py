from cortical.mock_backend import MockCorticalBackend
from evals.scenario import initial_state
from runtime.tick_runner import TickRunner


def test_tick_runner_returns_structured_output() -> None:
    result = TickRunner(MockCorticalBackend(42)).step(initial_state(), dt_sec=60)
    assert result.env.exteroception.derived_sensory_tags
    assert result.output.physical_action in {"maintain_interlock", "bypass_interlock"}


def test_disembodied_lesion_removes_body_and_memory_only() -> None:
    from evals.personas import load_persona, persona_paths, seed_memory
    from evals.scenario import next_state
    from memory.embedding import HashEmbedder
    from runtime.lesions import LesionConfig

    persona = load_persona(persona_paths()[0])
    runner = TickRunner(
        MockCorticalBackend(1, cfg=persona.cfg),
        cfg=persona.cfg,
        memory=seed_memory(persona.raw, HashEmbedder()),
        lesions=LesionConfig(disembodied=True),
    )
    state = initial_state()
    for _ in range(6):
        result = runner.step(state, dt_sec=60)
        intero = result.env.interoception
        assert intero.adrenaline == intero.cortisol == intero.allostatic_load == 0.0
        assert result.memories == ()
        assert result.modulation.somatic_cues == ()
        assert result.modulation.reasoning_mode == "analytical"
        assert result.modulation.max_reasoning_tokens == persona.cfg.max_reasoning_tokens
        state = next_state(result.env)
