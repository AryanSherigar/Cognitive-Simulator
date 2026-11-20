import argparse
import json
from pathlib import Path

from bridge.translator import build_modulation_signal
from cortical.llamacpp_backend import LlamaCppBackend
from evals.scenario import initial_state
from schemas import SimConfig


def run(model_path: Path) -> dict[str, str]:
    backend = LlamaCppBackend(str(model_path), n_gpu_layers=-1)
    env = initial_state()
    appraisal = backend.appraise(env)
    modulation = build_modulation_signal(appraisal, env.interoception, SimConfig())
    output = backend.formulate(env, modulation, ())
    consolidation = backend.consolidate(env, modulation, output)
    return {
        "appraisal": appraisal.model_dump_json(),
        "action": output.physical_action,
        "consolidation": consolidation.subjective_narrative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/cognitive-simulator.gguf")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.model_path), sort_keys=True))


if __name__ == "__main__":
    main()
