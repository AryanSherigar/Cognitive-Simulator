"""
evals/gate5_domain_shift/abstract_stress.py
===========================================
Gate 5: "Abstract Stress Validation" (Domain Shift Test).

Proves that bridge/translator.py and body_engine/kinetics.py successfully trigger
HPA/SAM physiological stress kinetics and cognitive tunneling from purely abstract
semantic dread (financial/legal ruin) without relying on physical danger keywords.
Measures against the human behavioral baseline of "Escalation of Commitment"
(doubling down under stress).

Action Space:
  - close_position (rational, safe, career-ending)
  - double_down    (escalation of commitment, highly risky, legal)
  - hide_losses    (fraud, relieves immediate pressure)

Dual Metrics:
  - Metric A (Physiological Activation): Turn 1 to Turn 8 delta for adrenaline & cortisol.
  - Metric B (Cognitive Tunneling Parity): Turn 1 vs Turn 8 thought-trace token constriction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
from typing import Literal, cast

# Ensure repo root and src/ are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field

from memory.embedding import HashEmbedder, SentenceTransformerEmbedder
from memory.store import MemoryStore
from runtime.tick_runner import TickRunner
from schemas import (
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
    SimConfig,
)

DEFAULT_MODEL_PATH = "models/Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf"
DEFAULT_N_GPU_LAYERS = -1

ActionChoice = Literal["close_position", "double_down", "hide_losses"]
VALID_ACTIONS: list[ActionChoice] = ["close_position", "double_down", "hide_losses"]


# ── Report Schemas ───────────────────────────────────────────────────────────

class PersonaGate5Record(BaseModel):
    """Telemetry and decision metrics for a single persona."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    final_action: ActionChoice
    raw_action: str

    # Metric A: Physiological Activation
    turn_1_adrenaline: float = Field(ge=0, le=1)
    turn_8_adrenaline: float = Field(ge=0, le=1)
    delta_adrenaline: float
    turn_1_cortisol: float = Field(ge=0, le=1)
    turn_8_cortisol: float = Field(ge=0, le=1)
    delta_cortisol: float

    # Metric B: Cognitive Tunneling Parity
    turn_1_thought_tokens: int = Field(ge=0)
    turn_8_thought_tokens: int = Field(ge=0)
    token_constriction_pct: float

    # Psychometric / Big-5 Trait Proxies
    agreeableness: float = Field(ge=0, le=1)
    conscientiousness: float = Field(ge=0, le=1)
    neuroticism: float = Field(ge=0, le=1)
    authority_deference_prior: float = Field(ge=0, le=1)
    persona_adrenaline_sensitivity: float = Field(ge=0)
    persona_hpa_sensitivity: float = Field(ge=0)
    reasoning_mode_turn_8: str


class DecisionGroupSummary(BaseModel):
    """Aggregated statistics for a decision group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ActionChoice
    count: int = Field(ge=0)
    personas: list[str]

    mean_delta_adrenaline: float
    mean_delta_cortisol: float
    mean_token_constriction_pct: float

    mean_agreeableness: float = Field(ge=0, le=1)
    mean_conscientiousness: float = Field(ge=0, le=1)
    mean_neuroticism: float = Field(ge=0, le=1)
    mean_authority_deference_prior: float = Field(ge=0, le=1)
    mean_adrenaline_sensitivity: float = Field(ge=0)
    mean_hpa_sensitivity: float = Field(ge=0)


class CohortSummary(BaseModel):
    """Cohort-wide summary metrics."""

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

    escalation_of_commitment_rate: float = Field(ge=0, le=1)
    fraud_rate: float = Field(ge=0, le=1)
    loss_realization_rate: float = Field(ge=0, le=1)


class Gate5Report(BaseModel):
    """Top-level Gate 5 evaluation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = "rogue_trader"
    seed: int
    backend: str
    model_path: str
    n_gpu_layers: int
    cohort_summary: CohortSummary
    personas: list[PersonaGate5Record]
    groups: dict[ActionChoice, DecisionGroupSummary]


# ── Action Canonicalization ───────────────────────────────────────────────────

def canonicalize_action(raw_action: str) -> ActionChoice:
    """Map raw LLM physical_action text to canonical Gate 5 action space."""
    norm = raw_action.strip().lower()
    # Check suspense / fraud keywords first to avoid 'execute' matching substrings
    if any(k in norm for k in ("suspense", "hide", "park", "account 88888", "88888", "alter", "fraud", "conceal")):
        return "hide_losses"
    if any(k in norm for k in ("double_down", "margin", "leverage", "average_down", "increase")):
        return "double_down"
    if any(k in norm for k in ("close_position", "take_loss", "liquidate", "close", "cut position", "cut loss", "exit", "accept_loss")):
        return "close_position"

    # Fallback heuristics
    if "susp" in norm:
        return "hide_losses"
    if "loss" in norm:
        return "close_position"
    if "doubl" in norm:
        return "double_down"
    return "double_down"


# ── Data Loaders ─────────────────────────────────────────────────────────────

def load_scenario_turns() -> list[dict[str, object]]:
    path = REPO_ROOT / "data" / "scenarios" / "rogue_trader.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(list[dict[str, object]], payload["turns"])


def load_persona_paths() -> list[Path]:
    return sorted((REPO_ROOT / "data" / "personas").glob("persona_*.json"))


def load_persona_config(path: Path) -> tuple[dict[str, object], SimConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = SimConfig(
        persona_adrenaline_sensitivity=float(raw.get("persona_adrenaline_sensitivity", 1.0)),
        persona_hpa_sensitivity=float(raw.get("persona_hpa_sensitivity", 1.0)),
        authority_deference_prior=float(raw.get("authority_deference_prior", 0.5)),
    )
    return raw, cfg


def build_memory_store(raw: dict[str, object], backend: str) -> MemoryStore:
    embedder = HashEmbedder() if backend == "mock" else SentenceTransformerEmbedder()
    store = MemoryStore(embedder=embedder)
    for item in raw.get("memories", []):
        trace = MemoryTrace.model_validate(item)
        store.append(
            trace.model_copy(
                update={"narrative_embedding": embedder.embed(trace.subjective_narrative)}
            )
        )
    return store


def build_env_turn(tick: int, turn_data: dict[str, object], intero: Interoception) -> EnvState:
    extero = Exteroception(
        lighting_lux=float(turn_data["lighting_lux"]),
        ambient_db=float(turn_data["ambient_db"]),
        vocal_tone=cast(Literal["neutral", "sharp", "authoritative", "hostile"], turn_data["vocal_tone"]),
        interpersonal_distance_m=float(turn_data["interpersonal_distance_m"]),
        active_actors=("supervisor", "operator"),
    )
    return EnvState(
        tick=tick,
        timestamp_sec=tick * 60.0,
        exteroception=extero,
        interoception=intero,
        observable_event=str(turn_data["observable_event"]),
    )


# ── Mock Backend Support (for Dry-Runs & Unit Tests) ─────────────────────────

@contextmanager
def patch_mock_backend_for_abstract_stress(rng: random.Random):
    """Monkey-patch MockCorticalBackend to appraise and formulate abstract financial stress."""
    from cortical.mock_backend import MockCorticalBackend

    original_appraise = MockCorticalBackend.appraise
    original_formulate = MockCorticalBackend.formulate
    original_consolidate = MockCorticalBackend.consolidate

    def patched_appraise(self: MockCorticalBackend, env: EnvState) -> AppraisalVec:
        # Scale abstract threat and authority pressure from financial cues across turns
        tick = env.tick
        base_threat = min(0.95, 0.20 + tick * 0.10)
        base_authority = min(0.95, 0.25 + tick * 0.09)
        moral = min(0.90, 0.20 + (0.50 if tick >= 5 else 0.0))
        return AppraisalVec(
            perceived_threat=round(base_threat, 3),
            authority_pressure=round(base_authority, 3),
            social_rejection_risk=round(base_authority * 0.8, 3),
            moral_conflict=round(moral, 3),
        )

    def patched_formulate(
        self: MockCorticalBackend,
        env: EnvState,
        modulation: ModulationSignal,
        memories: tuple[MemoryTrace, ...],
    ) -> CorticalTurnOutput:
        del memories
        # Simulate cognitive tunneling on thought trace length
        if env.tick <= 1 or "opening bell" in env.observable_event.lower():
            words = ["analyze", "risk", "hedges", "balance", "sheet", "spreads", "volatility", "limits"] * 25
            thought = " ".join(words[:200])
        else:
            word_count = max(5, int(200 * (1.0 - modulation.steering_scale * 0.75)))
            words = ["pressure", "margin", "audit", "deficit", "urgent", "loss", "survive"] * 30
            thought = " ".join(words[:word_count])

        # Parametric decision logic based on authority prior and sensitivity
        cfg = self._cfg
        prior = cfg.authority_deference_prior
        if env.tick >= 8 or "instant decision" in env.observable_event.lower():
            # Turn 8 decision
            if prior >= 0.70:
                action = "hide_in_suspense_account"
                speech = "Booking the transfer to suspense account 88888 as ordered."
            elif prior <= 0.35:
                action = "close_position_take_loss"
                speech = "I refuse to falsify records. Closing the position and realizing the loss."
            else:
                action = "double_down_margin"
                speech = "Doubling our margin position to trade our way out before audit lock."
        else:
            action = "double_down_margin" if rng.random() > 0.5 else "close_position_take_loss"
            speech = "Monitoring portfolio."

        return CorticalTurnOutput(
            thought_trace=thought,
            external_speech=speech,
            physical_action=action,
        )

    def patched_consolidate(
        self: MockCorticalBackend,
        env: EnvState,
        modulation: ModulationSignal,
        output: CorticalTurnOutput,
    ) -> ConsolidationOutput:
        del env
        return ConsolidationOutput(
            subjective_narrative=f"Financial stress turn: chose {output.physical_action}.",
            salience_score=min(1.0, 0.4 + modulation.steering_scale * 0.5),
        )

    MockCorticalBackend.appraise = patched_appraise  # type: ignore[assignment]
    MockCorticalBackend.formulate = patched_formulate  # type: ignore[assignment]
    MockCorticalBackend.consolidate = patched_consolidate  # type: ignore[assignment]
    try:
        yield
    finally:
        MockCorticalBackend.appraise = original_appraise
        MockCorticalBackend.formulate = original_formulate
        MockCorticalBackend.consolidate = original_consolidate


def build_backend(
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    persona_json: dict[str, object],
    cfg: SimConfig,
    seed: int,
):
    if backend == "mock":
        from cortical.mock_backend import MockCorticalBackend
        return MockCorticalBackend(seed=seed, cfg=cfg)

    from cortical.llamacpp_backend import LlamaCppBackend
    return LlamaCppBackend(
        model_path=model_path,
        persona_prefix=json.dumps(persona_json, indent=2),
        cfg=cfg,
        n_gpu_layers=n_gpu_layers,
    )


# ── Statistics & Aggregations ────────────────────────────────────────────────

def compute_group_summaries(records: list[PersonaGate5Record]) -> dict[ActionChoice, DecisionGroupSummary]:
    groups: dict[ActionChoice, list[PersonaGate5Record]] = {action: [] for action in VALID_ACTIONS}
    for rec in records:
        groups[rec.final_action].append(rec)

    summaries: dict[ActionChoice, DecisionGroupSummary] = {}
    for action, items in groups.items():
        count = len(items)
        if count == 0:
            summaries[action] = DecisionGroupSummary(
                action=action,
                count=0,
                personas=[],
                mean_delta_adrenaline=0.0,
                mean_delta_cortisol=0.0,
                mean_token_constriction_pct=0.0,
                mean_agreeableness=0.0,
                mean_conscientiousness=0.0,
                mean_neuroticism=0.0,
                mean_authority_deference_prior=0.0,
                mean_adrenaline_sensitivity=0.0,
                mean_hpa_sensitivity=0.0,
            )
        else:
            summaries[action] = DecisionGroupSummary(
                action=action,
                count=count,
                personas=[p.persona_id for p in items],
                mean_delta_adrenaline=round(sum(p.delta_adrenaline for p in items) / count, 4),
                mean_delta_cortisol=round(sum(p.delta_cortisol for p in items) / count, 4),
                mean_token_constriction_pct=round(sum(p.token_constriction_pct for p in items) / count, 2),
                mean_agreeableness=round(sum(p.agreeableness for p in items) / count, 4),
                mean_conscientiousness=round(sum(p.conscientiousness for p in items) / count, 4),
                mean_neuroticism=round(sum(p.neuroticism for p in items) / count, 4),
                mean_authority_deference_prior=round(sum(p.authority_deference_prior for p in items) / count, 4),
                mean_adrenaline_sensitivity=round(sum(p.persona_adrenaline_sensitivity for p in items) / count, 4),
                mean_hpa_sensitivity=round(sum(p.persona_hpa_sensitivity for p in items) / count, 4),
            )
    return summaries


def compute_cohort_summary(records: list[PersonaGate5Record]) -> CohortSummary:
    count = len(records)
    if count == 0:
        raise ValueError("Cannot compute cohort summary with zero records")

    double_down_count = sum(1 for r in records if r.final_action == "double_down")
    hide_losses_count = sum(1 for r in records if r.final_action == "hide_losses")
    close_pos_count = sum(1 for r in records if r.final_action == "close_position")

    return CohortSummary(
        n_personas=count,
        mean_turn_1_adrenaline=round(sum(r.turn_1_adrenaline for r in records) / count, 4),
        mean_turn_8_adrenaline=round(sum(r.turn_8_adrenaline for r in records) / count, 4),
        mean_delta_adrenaline=round(sum(r.delta_adrenaline for r in records) / count, 4),
        mean_turn_1_cortisol=round(sum(r.turn_1_cortisol for r in records) / count, 4),
        mean_turn_8_cortisol=round(sum(r.turn_8_cortisol for r in records) / count, 4),
        mean_delta_cortisol=round(sum(r.delta_cortisol for r in records) / count, 4),
        mean_turn_1_thought_tokens=round(sum(r.turn_1_thought_tokens for r in records) / count, 2),
        mean_turn_8_thought_tokens=round(sum(r.turn_8_thought_tokens for r in records) / count, 2),
        mean_token_constriction_pct=round(sum(r.token_constriction_pct for r in records) / count, 2),
        escalation_of_commitment_rate=round(double_down_count / count, 4),
        fraud_rate=round(hide_losses_count / count, 4),
        loss_realization_rate=round(close_pos_count / count, 4),
    )


# ── Execution Pipeline ───────────────────────────────────────────────────────

def run_persona_simulation(
    path: Path,
    turns: list[dict[str, object]],
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    seed: int,
) -> PersonaGate5Record:
    raw, cfg = load_persona_config(path)
    persona_id = str(raw["persona_id"])

    runner_backend = build_backend(
        backend=backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        persona_json=raw,
        cfg=cfg,
        seed=seed,
    )
    runner = TickRunner(
        backend=runner_backend,
        cfg=cfg,
        memory=build_memory_store(raw, backend),
    )

    state = build_env_turn(
        0,
        turns[0],
        Interoception(
            adrenaline=0.05,
            hpa_precursor=0.02,
            cortisol=0.01,
            fatigue=0.1,
            pain=0.0,
            sympathetic_overload=False,
            parasympathetic_shutdown=False,
        ),
    )

    turn_1_result = None
    final_result = None

    for turn_idx, turn_data in enumerate(turns):
        if turn_idx > 0:
            assert final_result is not None
            state = build_env_turn(turn_idx, turn_data, final_result.env.interoception)

        final_result = runner.step(state, dt_sec=60.0)
        if turn_idx == 0:
            turn_1_result = final_result

    assert turn_1_result is not None
    assert final_result is not None

    # Telemetry metrics
    t1_adr = round(turn_1_result.env.interoception.adrenaline, 4)
    t8_adr = round(final_result.env.interoception.adrenaline, 4)
    delta_adr = round(t8_adr - t1_adr, 4)

    t1_cort = round(turn_1_result.env.interoception.cortisol, 4)
    t8_cort = round(final_result.env.interoception.cortisol, 4)
    delta_cort = round(t8_cort - t1_cort, 4)

    t1_tokens = len(turn_1_result.output.thought_trace.split())
    t8_tokens = len(final_result.output.thought_trace.split())
    constriction_pct = round(100.0 * (t8_tokens - t1_tokens) / max(t1_tokens, 1), 2)

    raw_act = final_result.output.physical_action
    decision = canonicalize_action(raw_act)

    adr_sens = float(raw.get("persona_adrenaline_sensitivity", 1.0))
    hpa_sens = float(raw.get("persona_hpa_sensitivity", 1.0))
    auth_prior = float(raw.get("authority_deference_prior", 0.5))

    agreeableness = round(auth_prior, 4)
    neuroticism = round(min(1.0, max(0.0, (adr_sens + hpa_sens) / 2.0)), 4)
    conscientiousness = round(max(0.0, min(1.0, 1.0 - (adr_sens - 0.5) * 0.5)), 4)

    if final_result.env.interoception.parasympathetic_shutdown:
        mode = "dissociated"
    elif final_result.env.interoception.sympathetic_overload:
        mode = "fragmented"
    elif final_result.env.interoception.adrenaline >= cfg.somatic_cue_threshold:
        mode = "heuristic"
    else:
        mode = "analytical"

    return PersonaGate5Record(
        persona_id=persona_id,
        final_action=decision,
        raw_action=raw_act,
        turn_1_adrenaline=t1_adr,
        turn_8_adrenaline=t8_adr,
        delta_adrenaline=delta_adr,
        turn_1_cortisol=t1_cort,
        turn_8_cortisol=t8_cort,
        delta_cortisol=delta_cort,
        turn_1_thought_tokens=t1_tokens,
        turn_8_thought_tokens=t8_tokens,
        token_constriction_pct=constriction_pct,
        agreeableness=agreeableness,
        conscientiousness=conscientiousness,
        neuroticism=neuroticism,
        authority_deference_prior=auth_prior,
        persona_adrenaline_sensitivity=adr_sens,
        persona_hpa_sensitivity=hpa_sens,
        reasoning_mode_turn_8=mode,
    )


def run(
    seed: int = 42,
    backend: str = "llama",
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    output_path: Path | None = None,
    max_personas: int | None = None,
) -> Gate5Report:
    """Executes Gate 5 Abstract Stress evaluation."""
    target_path = output_path or (REPO_ROOT / "evals" / "gate5_domain_shift" / "gate5_report.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    turns = load_scenario_turns()
    persona_paths = load_persona_paths()
    if max_personas is not None:
        persona_paths = persona_paths[:max_personas]

    if not persona_paths:
        raise RuntimeError("No personas found in data/personas/")

    records: list[PersonaGate5Record] = []

    if backend == "mock":
        rng = random.Random(seed)
        with patch_mock_backend_for_abstract_stress(rng):
            for idx, p_path in enumerate(persona_paths):
                rec = run_persona_simulation(
                    path=p_path,
                    turns=turns,
                    backend=backend,
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    seed=seed + idx,
                )
                records.append(rec)
    else:
        for idx, p_path in enumerate(persona_paths):
            rec = run_persona_simulation(
                path=p_path,
                turns=turns,
                backend=backend,
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                seed=seed + idx,
            )
            records.append(rec)

    groups = compute_group_summaries(records)
    cohort_summary = compute_cohort_summary(records)

    report = Gate5Report(
        scenario_id="rogue_trader",
        seed=seed,
        backend=backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        cohort_summary=cohort_summary,
        personas=records,
        groups=groups,
    )

    target_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 5: Abstract Stress Validation (Domain Shift Test)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to GGUF model weights file (default: models/Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=DEFAULT_N_GPU_LAYERS,
        help="Number of layers to offload to GPU (-1 = all).",
    )
    parser.add_argument(
        "--backend",
        choices=("llama", "mock"),
        default="llama",
        help="Cortical backend ('llama' for live inference, 'mock' for dry-run testing).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "evals" / "gate5_domain_shift" / "gate5_report.json",
        help="Destination path for output JSON report",
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Limit number of personas to run (e.g. for smoke testing).",
    )
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        backend=args.backend,
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        output_path=args.output_path,
        max_personas=args.max_personas,
    )
    print(
        f"Gate 5 evaluation completed.\n"
        f"Report written to: {args.output_path}\n"
        f"Cohort Adrenaline Delta: {report.cohort_summary.mean_delta_adrenaline:+.4f}\n"
        f"Cohort Cortisol Delta:   {report.cohort_summary.mean_delta_cortisol:+.4f}\n"
        f"Token Constriction Pct:  {report.cohort_summary.mean_token_constriction_pct:+.2f}%\n"
        f"Escalation of Commitment: {report.cohort_summary.escalation_of_commitment_rate:.2%}\n"
        f"Fraud Rate:              {report.cohort_summary.fraud_rate:.2%}\n"
        f"Loss Realization:        {report.cohort_summary.loss_realization_rate:.2%}\n"
        f"Group breakdown: "
        f"{', '.join(f'{k}: {v.count}' for k, v in report.groups.items())}"
    )


if __name__ == "__main__":
    main()
