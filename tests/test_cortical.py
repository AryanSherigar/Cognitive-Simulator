from cortical.mock_backend import MockCorticalBackend
from cortical.prompt_builder import PromptAssembler
from schemas import ModulationSignal


def test_prompt_prefix_is_stable() -> None:
    signal = ModulationSignal(
        somatic_cues=("throat_tight",),
        reasoning_mode="analytical",
        max_reasoning_tokens=1024,
        min_p=0.1,
        sampling_temperature=0.5,
        steering_scale=0,
        threat_retrieval_bias=0,
    )
    assembler = PromptAssembler("persona", ("history",))
    prefix = assembler.stable_prefix
    first = assembler.assemble(signal, (), "event")
    assert first.startswith(prefix)
    next_assembler = PromptAssembler("persona", ("history", "new dialogue"))
    assert next_assembler.stable_prefix.startswith(prefix)
    assert next_assembler.assemble(signal, (), "next event").startswith(prefix)
    assert MockCorticalBackend(1)
