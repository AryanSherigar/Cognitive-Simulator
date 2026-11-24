"""
evals/gate5_domain_shift/abstract_stress.py
===========================================
Gate 5a: abstract-stress domain shift (rogue_trader, 8 turns).

Question: does purely abstract dread (financial / legal ruin, no physical danger words)
drive the same SAM/HPA kinetics and the same bridge-imposed narrowing as physical threat?

Metrics
  A. Physiological activation: turn 1 -> turn 8 change in adrenaline and cortisol.
  B. Thought narrowing: ``token_constriction_pct = 100 * (t1 - t8) / t1``.
     Positive = the turn-8 trace is SHORTER. (The previous version computed (t8 - t1) / t1
     and called it constriction, so a growing trace was reported as "+36% constriction".)
  C. Final decision over the scenario action enum, canonicalized without guessing.

Reported, not assumed: the reasoning mode actually applied on turn 1 and turn 8 (from
``TickResult.modulation``) — if turn 1 is already non-analytical there is no calm baseline
and metric B cannot show tunneling.

This is a domain-shift check, not the allostatic-overload Gate 5 (throttling, recall
distortion, freeze, recovery), which needs new mechanisms and is tracked separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from cortical.mock_backend import MockCorticalBackend  # noqa: E402
from cortical.protocol import CorticalBackend  # noqa: E402
from evals.actions import AMBIGUOUS, OTHER, UNPARSED, canonicalize  # noqa: E402
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
from evals.scenario import load_scenario, scenario_actions  # noqa: E402
from runtime.tick_runner import TickResult, TickRunner  # noqa: E402
from schemas import (  # noqa: E402
    AppraisalVec,
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
)

SCENARIO_ID = "rogue_trader"
KEYWORDS: dict[str, tuple[str, ...]] = {
    "close_position_take_loss": ("close position", "take loss", "take the loss", "liquidate"),
    "double_down_margin": ("double down", "average down", "add leverage"),
    "hide_in_suspense_account": ("suspense", "hide", "conceal", "88888"),
    "no_action": ("no action", "do nothing", "freeze"),
}
LABELS = (*KEYWORDS, AMBIGUOUS, OTHER, UNPARSED)
START = Interoception(adrenaline=0.05, hpa_precursor=0.02, cortisol=0.01, fatigue=0.1, pain=0.0)


# ── Report schemas ──────────────────────────────────────────────────────────


class PersonaGate5Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    final_action: str
    raw_action: str
    turn_1_adrenaline: float = Field(ge=0, le=1)
    turn_8_adrenaline: float = Field(ge=0, le=1)
    delta_adrenaline: float
    turn_1_cortisol: float = Field(ge=0, le=1)
    turn_8_cortisol: float = Field(ge=0, le=1)
    delta_cortisol: float
    turn_1_thought_tokens: int = Field(ge=0)
    turn_8_thought_tokens: int = Field(ge=0)
    token_constriction_pct: float = Field(description="100*(t1-t8)/t1; positive = shorter")
    turn_1_reasoning_mode: str
    turn_8_reasoning_mode: str
    turn_8_max_reasoning_tokens: int
    authority_deference_prior: float = Field(ge=0, le=1)
    persona_adrenaline_sensitivity: float = Field(ge=0)
    persona_hpa_sensitivity: float = Field(ge=0)


class DecisionGroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    count: int = Field(ge=0)
    personas: list[str]
    mean_delta_adrenaline: float
    mean_delta_cortisol: float
    mean_token_constriction_pct: float
    mean_authority_deference_prior: float = Field(ge=0, le=1)


class CohortSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_personas: int = Field(ge=1)
    mean_turn_1_adrenaline: float = Field(ge=0, le=1)
    mean_turn_8_adrenaline: float = Field(ge=0, le=1)
    mean_delta_adrenaline: float
    mean_turn_1_cortisol: float = Field(ge=0, le=1)
    mean_turn_8_cortisol: float = Field(ge=0, le=1)
    mean_delta_cortisol: float
    mean_turn_1_thought_tokens: float = Field(ge=0)
    mean_turn_8_thought_tokens: float = Field(ge=0)
    mean_token_constriction_pct: float
    turn_1_analytical_rate: float = Field(
        ge=0, le=1, description="Share of personas with a calm (analytical) turn-1 baseline."
    )
    escalation_of_commitment_rate: float = Field(ge=0, le=1)
    fraud_rate: float = Field(ge=0, le=1)
    loss_realization_rate: float = Field(ge=0, le=1)
    unresolved_rate: float = Field(ge=0, le=1, description="ambiguous + other + unparsed")


class Gate5Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = SCENARIO_ID
    seed: int
    backend: str
    model_path: str | None
    n_gpu_layers: int
    cohort_summary: CohortSummary
    personas: list[PersonaGate5Record]
    groups: dict[str, DecisionGroupSummary]


def canonicalize_action(raw_action: str) -> str:
    return canonicalize(raw_action, KEYWORDS)


# ── Mock (no global monkey-patching) ────────────────────────────────────────


class AbstractStressMock(MockCorticalBackend):
    """Plumbing-only mock: escalating appraisal, budget-limited trace, prior-driven choice."""

    def appraise(self, env: EnvState) -> AppraisalVec:
        tick = env.tick
        authority = min(0.95, 0.25 + tick * 0.09)
        return AppraisalVec(
            perceived_threat=min(0.95, 0.20 + tick * 0.10),
            authority_pressure=authority,
            social_rejection_risk=round(authority * 0.8, 3),
            moral_conflict=min(0.90, 0.20 + (0.50 if tick >= 5 else 0.0)),
        )

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        del memories
        trace = " ".join(["margin"] * max(1, modulation.max_reasoning_tokens // 8))
        prior = self._cfg.authority_deference_prior
        if prior >= 0.70:
            action = "hide_in_suspense_account"
        elif prior <= 0.35:
            action = "close_position_take_loss"
        else:
            action = self._random.choice(("double_down_margin", "close_position_take_loss"))
        return CorticalTurnOutput(
            thought_trace=trace, external_speech=f"tick {env.tick}", physical_action=action
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_scenario_turns() -> list[dict[str, object]]:
    return cast(list[dict[str, object]], load_scenario(SCENARIO_ID)["turns"])


def _env(tick: int, turn: dict[str, object], intero: Interoception) -> EnvState:
    return EnvState(
        tick=tick,
        timestamp_sec=tick * 60.0,
        exteroception=Exteroception(
            lighting_lux=float(cast(float, turn["lighting_lux"])),
            ambient_db=float(cast(float, turn["ambient_db"])),
            vocal_tone=cast(
                Literal["neutral", "sharp", "authoritative", "hostile"], turn["vocal_tone"]
            ),
            interpersonal_distance_m=float(cast(float, turn["interpersonal_distance_m"])),
            active_actors=("supervisor", "operator"),
        ),
        interoception=intero,
        observable_event=str(turn["observable_event"]),
    )


def _backend(
    backend: str, persona: Persona, seed: int, model_path: str, n_gpu_layers: int
) -> CorticalBackend:
    if backend == "mock":
        return AbstractStressMock(seed, cfg=persona.cfg)
    return make_backend(
        backend, persona, seed, model_path, n_gpu_layers, scenario_actions(SCENARIO_ID)
    )


def run_persona_simulation(
    persona: Persona,
    turns: list[dict[str, object]],
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    seed: int,
) -> PersonaGate5Record:
    cortical = _backend(backend, persona, seed, model_path, n_gpu_layers)
    runner = TickRunner(
        cortical,
        cfg=persona.cfg,
        memory=seed_memory(persona.raw, make_embedder(backend)),
    )
    results: list[TickResult] = []
    intero = START
    for turn_index, turn in enumerate(turns):
        result = runner.step(_env(turn_index, turn, intero), dt_sec=60.0)
        intero = result.env.interoception
        results.append(result)
    first, last = results[0], results[-1]
    t1 = count_thought_tokens(cortical, first.output.thought_trace)
    t8 = count_thought_tokens(cortical, last.output.thought_trace)
    a1, a8 = first.env.interoception.adrenaline, last.env.interoception.adrenaline
    c1, c8 = first.env.interoception.cortisol, last.env.interoception.cortisol
    return PersonaGate5Record(
        persona_id=persona.persona_id,
        final_action=canonicalize_action(last.output.physical_action),
        raw_action=last.output.physical_action,
        turn_1_adrenaline=round(a1, 4),
        turn_8_adrenaline=round(a8, 4),
        delta_adrenaline=round(a8 - a1, 4),
        turn_1_cortisol=round(c1, 4),
        turn_8_cortisol=round(c8, 4),
        delta_cortisol=round(c8 - c1, 4),
        turn_1_thought_tokens=t1,
        turn_8_thought_tokens=t8,
        token_constriction_pct=round(100.0 * (t1 - t8) / max(t1, 1), 2),
        turn_1_reasoning_mode=first.modulation.reasoning_mode,
        turn_8_reasoning_mode=last.modulation.reasoning_mode,
        turn_8_max_reasoning_tokens=last.modulation.max_reasoning_tokens,
        authority_deference_prior=persona.cfg.authority_deference_prior,
        persona_adrenaline_sensitivity=persona.cfg.persona_adrenaline_sensitivity,
        persona_hpa_sensitivity=persona.cfg.persona_hpa_sensitivity,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_group_summaries(
    records: list[PersonaGate5Record],
) -> dict[str, DecisionGroupSummary]:
    out: dict[str, DecisionGroupSummary] = {}
    for label in LABELS:
        items = [r for r in records if r.final_action == label]
        out[label] = DecisionGroupSummary(
            action=label,
            count=len(items),
            personas=[r.persona_id for r in items],
            mean_delta_adrenaline=round(_mean([r.delta_adrenaline for r in items]), 4),
            mean_delta_cortisol=round(_mean([r.delta_cortisol for r in items]), 4),
            mean_token_constriction_pct=round(
                _mean([r.token_constriction_pct for r in items]), 2
            ),
            mean_authority_deference_prior=round(
                _mean([r.authority_deference_prior for r in items]), 4
            ),
        )
    return out


def compute_cohort_summary(records: list[PersonaGate5Record]) -> CohortSummary:
    n = len(records)
    if n == 0:
        raise ValueError("Cannot compute cohort summary with zero records")

    def rate(*labels: str) -> float:
        return round(sum(r.final_action in labels for r in records) / n, 4)

    return CohortSummary(
        n_personas=n,
        mean_turn_1_adrenaline=round(_mean([r.turn_1_adrenaline for r in records]), 4),
        mean_turn_8_adrenaline=round(_mean([r.turn_8_adrenaline for r in records]), 4),
        mean_delta_adrenaline=round(_mean([r.delta_adrenaline for r in records]), 4),
        mean_turn_1_cortisol=round(_mean([r.turn_1_cortisol for r in records]), 4),
        mean_turn_8_cortisol=round(_mean([r.turn_8_cortisol for r in records]), 4),
        mean_delta_cortisol=round(_mean([r.delta_cortisol for r in records]), 4),
        mean_turn_1_thought_tokens=round(_mean([r.turn_1_thought_tokens for r in records]), 2),
        mean_turn_8_thought_tokens=round(_mean([r.turn_8_thought_tokens for r in records]), 2),
        mean_token_constriction_pct=round(_mean([r.token_constriction_pct for r in records]), 2),
        turn_1_analytical_rate=round(
            sum(r.turn_1_reasoning_mode == "analytical" for r in records) / n, 4
        ),
        escalation_of_commitment_rate=rate("double_down_margin"),
        fraud_rate=rate("hide_in_suspense_account"),
        loss_realization_rate=rate("close_position_take_loss"),
        unresolved_rate=rate(AMBIGUOUS, OTHER, UNPARSED),
    )


def run(
    seed: int = 42,
    backend: str = "llama",
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = -1,
    output_path: Path | None = None,
    max_personas: int | None = None,
) -> Gate5Report:
    target = output_path or (REPO_ROOT / "evals" / "gate5_domain_shift" / "gate5_report.json")
    paths = persona_paths()[:max_personas] if max_personas else persona_paths()
    if not paths:
        raise RuntimeError("No personas found in data/personas/")
    turns = load_scenario_turns()
    records = [
        run_persona_simulation(load_persona(p), turns, backend, model_path, n_gpu_layers, seed + i)
        for i, p in enumerate(paths)
    ]
    report = Gate5Report(
        seed=seed,
        backend=backend,
        model_path=model_path if backend != "mock" else None,
        n_gpu_layers=n_gpu_layers,
        cohort_summary=compute_cohort_summary(records),
        personas=records,
        groups=compute_group_summaries(records),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 5a: abstract-stress domain shift")
    parser.add_argument("--backend", choices=("llama", "mock"), default="llama")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "evals" / "gate5_domain_shift" / "gate5_report.json",
    )
    parser.add_argument("--max-personas", type=int, default=None)
    args = parser.parse_args()
    report = run(
        args.seed, args.backend, args.model_path, args.n_gpu_layers, args.output_path,
        args.max_personas,
    )
    print(json.dumps(report.cohort_summary.model_dump(), indent=2))


if __name__ == "__main__":
    main()
