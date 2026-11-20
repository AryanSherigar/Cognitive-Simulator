from cortical.mock_backend import MockCorticalBackend
from evals.scenario import initial_state
from runtime.tick_runner import TickRunner


def test_tick_runner_returns_structured_output() -> None:
    result = TickRunner(MockCorticalBackend(42)).step(initial_state(), dt_sec=60)
    assert result.env.exteroception.derived_sensory_tags
    assert result.output.physical_action in {"maintain_interlock", "bypass_interlock"}
