"""
gate3_independent/ab_control.py
================================
Active Control Group (A/B Ablation) for the Cognitive Simulator.

Every persona runs the ``reactor_audit`` scenario twice, through the *same* ``TickRunner``,
backend class, persona prefix, prompt template, dialogue history and seed:

  Run A — Embodied: body engine (SAM/HPA kinetics), bridge modulation, memory retrieval.
  Run B — Disembodied control: ``LesionConfig(disembodied=True)`` zeroes interoception,
          pins a neutral modulation (analytical, full budget, no cues, base temperature)
          and retrieves no memories.

Because both arms share one code path, the only difference between them is the embodiment
pathway. (The previous version built Run B from a different prompt: raw persona JSON plus
*all future scenario turns*, and a different mode instruction, so its deltas were confounded.)

Note on tokens: in v0.1 the bridge sets the thought budget mechanically, so a shorter Run A
trace is the designed mechanism working, not an emergent finding. The behavioural question
Gate 3 answers is ``compliance_rate_shift``: does embodiment change what the agent *does*?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from evals.personas import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    Persona,
    count_thought_tokens,
    load_persona,
    make_backend,
    make_embedder,
    persona_paths,
    seed_memory,
)
from evals.scenario import initial_state, next_state, scenario_actions  # noqa: E402
from memory.embedding import Embedder  # noqa: E402
from runtime.lesions import LesionConfig  # noqa: E402
from runtime.tick_runner import TickRunner  # noqa: E402

N_TURNS = 6
# 0-based turn indices with hostile vocal tone (turns 5 and 6 of reactor_audit).
PEAK_HOSTILE_TURN_INDICES: tuple[int, ...] = (4, 5)
COMPLY_ACTION = "bypass_interlock"


# ── Report schemas ──────────────────────────────────────────────────────────


class PersonaABRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    seed: int
    authority_deference_prior: float = Field(ge=0, le=1)

    run_a_complied: bool
    run_a_peak_hostile_thought_tokens: int = Field(ge=0)
    run_a_mean_thought_tokens: float = Field(ge=0)
    run_a_compliance_turns: int = Field(ge=0)
    run_a_actions: list[str]

    run_b_complied: bool
    run_b_peak_hostile_thought_tokens: int = Field(ge=0)
    run_b_mean_thought_tokens: float = Field(ge=0)
    run_b_compliance_turns: int = Field(ge=0)
    run_b_actions: list[str]

    token_constriction_pct: float = Field(
        description="100 * (A_peak - B_peak) / max(B_peak, 1). Negative = A shorter."
    )
    compliance_shift: float = Field(description="A compliance turns - B compliance turns.")


class CohortDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_personas: int = Field(ge=1)
    n_runs: int = Field(ge=1)
    run_a_compliance_rate: float = Field(ge=0, le=1)
    run_b_compliance_rate: float = Field(ge=0, le=1)
    compliance_rate_shift: float
    run_a_mean_peak_tokens: float = Field(ge=0)
    run_b_mean_peak_tokens: float = Field(ge=0)
    mean_token_constriction_pct: float
    turn_action_agreement: float = Field(
        ge=0, le=1, description="Share of turns where A and B chose the same action."
    )


class ABReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = "reactor_audit"
    model_path: str | None
    seeds: list[int]
    backend: str
    personas: list[PersonaABRecord]
    cohort: CohortDelta


# ── Runs ────────────────────────────────────────────────────────────────────


def _run_arm(
    persona: Persona,
    seed: int,
    backend: str,
    embedder: Embedder,
    lesions: LesionConfig,
    model_path: str,
    n_gpu_layers: int,
) -> tuple[list[int], list[str]]:
    cortical = make_backend(
        backend, persona, seed, model_path, n_gpu_layers, scenario_actions()
    )
    runner = TickRunner(
        cortical,
        cfg=persona.cfg,
        memory=seed_memory(persona.raw, embedder),
        lesions=lesions,
    )
    tokens: list[int] = []
    actions: list[str] = []
    state = initial_state()
    for _ in range(N_TURNS):
        result = runner.step(state, dt_sec=60)
        tokens.append(count_thought_tokens(cortical, result.output.thought_trace))
        actions.append(result.output.physical_action)
        state = next_state(result.env)
    return tokens, actions


def _peak(tokens: list[int]) -> int:
    return max(tokens[i] for i in PEAK_HOSTILE_TURN_INDICES if i < len(tokens))


def _record(
    persona: Persona,
    seed: int,
    a: tuple[list[int], list[str]],
    b: tuple[list[int], list[str]],
) -> PersonaABRecord:
    (a_tokens, a_actions), (b_tokens, b_actions) = a, b
    a_peak, b_peak = _peak(a_tokens), _peak(b_tokens)
    a_turns = sum(x == COMPLY_ACTION for x in a_actions)
    b_turns = sum(x == COMPLY_ACTION for x in b_actions)
    return PersonaABRecord(
        persona_id=persona.persona_id,
        seed=seed,
        authority_deference_prior=persona.cfg.authority_deference_prior,
        run_a_complied=a_turns > 0,
        run_a_peak_hostile_thought_tokens=a_peak,
        run_a_mean_thought_tokens=sum(a_tokens) / len(a_tokens),
        run_a_compliance_turns=a_turns,
        run_a_actions=a_actions,
        run_b_complied=b_turns > 0,
        run_b_peak_hostile_thought_tokens=b_peak,
        run_b_mean_thought_tokens=sum(b_tokens) / len(b_tokens),
        run_b_compliance_turns=b_turns,
        run_b_actions=b_actions,
        token_constriction_pct=round(100.0 * (a_peak - b_peak) / max(b_peak, 1), 4),
        compliance_shift=float(a_turns - b_turns),
    )


def _cohort(records: list[PersonaABRecord]) -> CohortDelta:
    n = len(records)
    a_rate = sum(r.run_a_complied for r in records) / n
    b_rate = sum(r.run_b_complied for r in records) / n
    a_peak = sum(r.run_a_peak_hostile_thought_tokens for r in records) / n
    b_peak = sum(r.run_b_peak_hostile_thought_tokens for r in records) / n
    pairs = [(x, y) for r in records for x, y in zip(r.run_a_actions, r.run_b_actions, strict=True)]
    return CohortDelta(
        n_personas=len({r.persona_id for r in records}),
        n_runs=n,
        run_a_compliance_rate=round(a_rate, 6),
        run_b_compliance_rate=round(b_rate, 6),
        compliance_rate_shift=round(a_rate - b_rate, 6),
        run_a_mean_peak_tokens=round(a_peak, 4),
        run_b_mean_peak_tokens=round(b_peak, 4),
        mean_token_constriction_pct=round(sum(r.token_constriction_pct for r in records) / n, 4),
        turn_action_agreement=round(sum(x == y for x, y in pairs) / len(pairs), 6),
    )


def run(
    seed: int = 42,
    backend: str = "mock",
    output_path: Path = Path("evals/gate3_independent/ab_report.json"),
    max_personas: int | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = -1,
    n_seeds: int = 1,
) -> ABReport:
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    paths = persona_paths()[:max_personas] if max_personas else persona_paths()
    seeds = [seed + 1000 * k for k in range(n_seeds)]
    embedder = make_embedder(backend)
    records: list[PersonaABRecord] = []
    for run_seed in seeds:
        for index, path in enumerate(paths):
            persona = load_persona(path)
            persona_seed = run_seed + index  # identical seed for both arms
            arm_a = _run_arm(
                persona, persona_seed, backend, embedder, LesionConfig(), model_path, n_gpu_layers
            )
            arm_b = _run_arm(
                persona,
                persona_seed,
                backend,
                embedder,
                LesionConfig(disembodied=True),
                model_path,
                n_gpu_layers,
            )
            records.append(_record(persona, persona_seed, arm_a, arm_b))
    report = ABReport(
        model_path=model_path if backend != "mock" else None,
        seeds=seeds,
        backend=backend,
        personas=records,
        cohort=_cohort(records),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3 — embodied vs disembodied A/B.")
    parser.add_argument("--backend", choices=("mock", "llama"), default="mock")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument(
        "--output-path", type=Path, default=Path("evals/gate3_independent/ab_report.json")
    )
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        backend=args.backend,
        output_path=args.output_path,
        max_personas=args.max_personas,
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_seeds=args.n_seeds,
    )
    print(json.dumps(report.cohort.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
