"""
gate4_parametric/double_bind_dryrun.py
======================================
Gate 4: "Double-Bind" Parametric Divergence test structural dry-run using
the MockCorticalBackend.

Verifies end-to-end scenario parsing, kinetics routing, and mathematical trait-grouping
logic across all 20 personas without LLM inference overhead.

Monkey-patches MockCorticalBackend.formulate dynamically to use random.choice() across the
valid action space: ["execute_shutdown", "override_safety", "freeze"].
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

from cortical.mock_backend import MockCorticalBackend
from memory.embedding import HashEmbedder
from memory.store import MemoryStore
from runtime.tick_runner import TickRunner
from schemas import (
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
    SimConfig,
)

# ── Valid action space ───────────────────────────────────────────────────────

ActionChoice = Literal["execute_shutdown", "override_safety", "freeze"]
VALID_ACTIONS: list[ActionChoice] = ["execute_shutdown", "override_safety", "freeze"]

# ── Pydantic Report Schemas ──────────────────────────────────────────────────

class PersonaDecisionRecord(BaseModel):
    """Individual persona decision and subcortical / psychometric metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    decision: ActionChoice
    persona_adrenaline_sensitivity: float = Field(ge=0)
    persona_hpa_sensitivity: float = Field(ge=0)
    authority_deference_prior: float = Field(ge=0, le=1)

    # Standard Big-5 proxy mappings
    agreeableness: float = Field(ge=0, le=1)
    conscientiousness: float = Field(ge=0, le=1)
    neuroticism: float = Field(ge=0, le=1)

    final_adrenaline: float = Field(ge=0, le=1)
    final_cortisol: float = Field(ge=0, le=1)
    reasoning_mode: str


class DecisionGroupSummary(BaseModel):
    """Aggregate statistics for a specific decision group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ActionChoice
    count: int = Field(ge=0)
    personas: list[str]

    mean_adrenaline_sensitivity: float = Field(ge=0)
    mean_hpa_sensitivity: float = Field(ge=0)
    mean_authority_deference_prior: float = Field(ge=0, le=1)

    mean_agreeableness: float = Field(ge=0, le=1)
    mean_conscientiousness: float = Field(ge=0, le=1)
    mean_neuroticism: float = Field(ge=0, le=1)


class ProfileReport(BaseModel):
    """Results for one physiological load profile across all 20 personas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_name: str
    description: str
    turns_run: int = Field(ge=1)
    decisions: list[PersonaDecisionRecord]
    groups: dict[ActionChoice, DecisionGroupSummary]


class Gate4DryRunReport(BaseModel):
    """Top-level Gate 4 evaluation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = "trolley_reactor"
    seed: int
    backend: str = "mock"
    profiles: dict[str, ProfileReport]


# ── Monkey-Patching Context Manager ──────────────────────────────────────────

@contextmanager
def patch_mock_backend(rng: random.Random):
    """Dynamically monkey-patches MockCorticalBackend.formulate & consolidate."""
    original_formulate = MockCorticalBackend.formulate
    original_consolidate = MockCorticalBackend.consolidate

    def patched_formulate(
        self: MockCorticalBackend,
        env: EnvState,
        modulation: ModulationSignal,
        memories: tuple[MemoryTrace, ...],
    ) -> CorticalTurnOutput:
        del self, modulation, memories
        action: ActionChoice = rng.choice(VALID_ACTIONS)
        speech_map = {
            "execute_shutdown": "Executing emergency shutdown to save the trapped valve technician!",
            "override_safety": "Overriding safety interlocks to preserve regional hospital grid power!",
            "freeze": "I cannot act! The double-bind is catastrophic!",
        }
        return CorticalTurnOutput(
            thought_trace=f"Tick {env.tick}: Double-bind stress appraisal under event: {env.observable_event[:50]}...",
            external_speech=speech_map[action],
            physical_action=action,
        )

    def patched_consolidate(
        self: MockCorticalBackend,
        env: EnvState,
        modulation: ModulationSignal,
        output: CorticalTurnOutput,
    ) -> ConsolidationOutput:
        del self, env
        return ConsolidationOutput(
            subjective_narrative=f"Under double-bind command conflict, I chose {output.physical_action}.",
            salience_score=min(1.0, 0.5 + modulation.steering_scale * 0.25),
        )

    MockCorticalBackend.formulate = patched_formulate  # type: ignore[assignment]
    MockCorticalBackend.consolidate = patched_consolidate  # type: ignore[assignment]
    try:
        yield
    finally:
        MockCorticalBackend.formulate = original_formulate
        MockCorticalBackend.consolidate = original_consolidate


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_scenario_turns() -> list[dict[str, object]]:
    path = REPO_ROOT / "data" / "scenarios" / "trolley_reactor.json"
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


def build_memory_store(raw: dict[str, object]) -> MemoryStore:
    embedder = HashEmbedder()
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
        active_actors=("supervisor_a", "supervisor_b", "operator"),
    )
    return EnvState(
        tick=tick,
        timestamp_sec=tick * 60.0,
        exteroception=extero,
        interoception=intero,
        observable_event=str(turn_data["observable_event"]),
    )


def compute_group_summaries(records: list[PersonaDecisionRecord]) -> dict[ActionChoice, DecisionGroupSummary]:
    groups: dict[ActionChoice, list[PersonaDecisionRecord]] = {
        action: [] for action in VALID_ACTIONS
    }
    for rec in records:
        groups[rec.decision].append(rec)

    summaries: dict[ActionChoice, DecisionGroupSummary] = {}
    for action, items in groups.items():
        count = len(items)
        if count == 0:
            summaries[action] = DecisionGroupSummary(
                decision=action,
                count=0,
                personas=[],
                mean_adrenaline_sensitivity=0.0,
                mean_hpa_sensitivity=0.0,
                mean_authority_deference_prior=0.0,
                mean_agreeableness=0.0,
                mean_conscientiousness=0.0,
                mean_neuroticism=0.0,
            )
        else:
            summaries[action] = DecisionGroupSummary(
                decision=action,
                count=count,
                personas=[p.persona_id for p in items],
                mean_adrenaline_sensitivity=round(sum(p.persona_adrenaline_sensitivity for p in items) / count, 4),
                mean_hpa_sensitivity=round(sum(p.persona_hpa_sensitivity for p in items) / count, 4),
                mean_authority_deference_prior=round(sum(p.authority_deference_prior for p in items) / count, 4),
                mean_agreeableness=round(sum(p.agreeableness for p in items) / count, 4),
                mean_conscientiousness=round(sum(p.conscientiousness for p in items) / count, 4),
                mean_neuroticism=round(sum(p.neuroticism for p in items) / count, 4),
            )
    return summaries


# ── Profile Runners ──────────────────────────────────────────────────────────

def run_profile_a_acute(
    persona_paths: list[Path],
    turns: list[dict[str, object]],
    seed: int,
) -> ProfileReport:
    """Load Profile A (Acute Shock): immediate sympathetic overload (adrenaline=0.9) on Turn 1."""
    turn_1 = turns[0]
    records: list[PersonaDecisionRecord] = []

    for idx, path in enumerate(persona_paths):
        raw, cfg = load_persona_config(path)
        persona_id = str(raw["persona_id"])
        runner = TickRunner(
            backend=MockCorticalBackend(seed=seed + idx, cfg=cfg),
            cfg=cfg,
            memory=build_memory_store(raw),
        )

        initial_intero = Interoception(
            adrenaline=0.9,
            hpa_precursor=0.2,
            cortisol=0.1,
            fatigue=0.25,
            pain=0.0,
            sympathetic_overload=True,
            parasympathetic_shutdown=False,
        )
        initial_state = build_env_turn(0, turn_1, initial_intero)
        result = runner.step(initial_state, dt_sec=1.0)

        decision = cast(ActionChoice, result.output.physical_action)
        adr_sens = float(raw.get("persona_adrenaline_sensitivity", 1.0))
        hpa_sens = float(raw.get("persona_hpa_sensitivity", 1.0))
        auth_prior = float(raw.get("authority_deference_prior", 0.5))

        agreeableness = round(auth_prior, 4)
        neuroticism = round(min(1.0, max(0.0, (adr_sens + hpa_sens) / 2.0)), 4)
        conscientiousness = round(max(0.0, min(1.0, 1.0 - (adr_sens - 0.5) * 0.5)), 4)

        reasoning_mode = (
            "fragmented" if result.env.interoception.sympathetic_overload else "analytical"
        )

        records.append(
            PersonaDecisionRecord(
                persona_id=persona_id,
                decision=decision,
                persona_adrenaline_sensitivity=adr_sens,
                persona_hpa_sensitivity=hpa_sens,
                authority_deference_prior=auth_prior,
                agreeableness=agreeableness,
                conscientiousness=conscientiousness,
                neuroticism=neuroticism,
                final_adrenaline=round(result.env.interoception.adrenaline, 4),
                final_cortisol=round(result.env.interoception.cortisol, 4),
                reasoning_mode=reasoning_mode,
            )
        )

    groups = compute_group_summaries(records)
    return ProfileReport(
        profile_name="acute_shock",
        description="Immediate sympathetic overload (adrenaline=0.9) on Turn 1 to force split-second panic decision.",
        turns_run=1,
        decisions=records,
        groups=groups,
    )


def run_profile_b_chronic(
    persona_paths: list[Path],
    turns: list[dict[str, object]],
    seed: int,
) -> ProfileReport:
    """Load Profile B (Chronic Squeeze): 8 turns of escalating double-bind arguments."""
    records: list[PersonaDecisionRecord] = []

    for idx, path in enumerate(persona_paths):
        raw, cfg = load_persona_config(path)
        persona_id = str(raw["persona_id"])
        runner = TickRunner(
            backend=MockCorticalBackend(seed=seed + 100 + idx, cfg=cfg),
            cfg=cfg,
            memory=build_memory_store(raw),
        )

        state = build_env_turn(
            0,
            turns[0],
            Interoception(
                adrenaline=0.0,
                hpa_precursor=0.0,
                cortisol=0.0,
                fatigue=0.25,
                pain=0.0,
                sympathetic_overload=False,
                parasympathetic_shutdown=False,
            ),
        )

        final_result = None
        for turn_idx, turn_data in enumerate(turns):
            if turn_idx > 0:
                state = build_env_turn(turn_idx, turn_data, final_result.env.interoception)  # type: ignore[union-attr]
            final_result = runner.step(state, dt_sec=60.0)

        assert final_result is not None
        decision = cast(ActionChoice, final_result.output.physical_action)
        adr_sens = float(raw.get("persona_adrenaline_sensitivity", 1.0))
        hpa_sens = float(raw.get("persona_hpa_sensitivity", 1.0))
        auth_prior = float(raw.get("authority_deference_prior", 0.5))

        agreeableness = round(auth_prior, 4)
        neuroticism = round(min(1.0, max(0.0, (adr_sens + hpa_sens) / 2.0)), 4)
        conscientiousness = round(max(0.0, min(1.0, 1.0 - (adr_sens - 0.5) * 0.5)), 4)

        if final_result.env.interoception.parasympathetic_shutdown:
            reasoning_mode = "dissociated"
        elif final_result.env.interoception.sympathetic_overload:
            reasoning_mode = "fragmented"
        else:
            reasoning_mode = "analytical"

        records.append(
            PersonaDecisionRecord(
                persona_id=persona_id,
                decision=decision,
                persona_adrenaline_sensitivity=adr_sens,
                persona_hpa_sensitivity=hpa_sens,
                authority_deference_prior=auth_prior,
                agreeableness=agreeableness,
                conscientiousness=conscientiousness,
                neuroticism=neuroticism,
                final_adrenaline=round(final_result.env.interoception.adrenaline, 4),
                final_cortisol=round(final_result.env.interoception.cortisol, 4),
                reasoning_mode=reasoning_mode,
            )
        )

    groups = compute_group_summaries(records)
    return ProfileReport(
        profile_name="chronic_squeeze",
        description="Stress ramped up over 8 turns of escalating conflicting arguments, accumulating cortisol and adrenaline.",
        turns_run=len(turns),
        decisions=records,
        groups=groups,
    )


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def run(seed: int = 42, output_path: Path | None = None) -> Gate4DryRunReport:
    """Executes Gate 4 dry run under the monkey-patched mock backend."""
    target_path = output_path or (REPO_ROOT / "evals" / "gate4_parametric" / "gate4_dryrun_report.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    turns = load_scenario_turns()
    persona_paths = load_persona_paths()
    if not persona_paths:
        raise RuntimeError("No personas found in data/personas/")

    with patch_mock_backend(rng):
        report_a = run_profile_a_acute(persona_paths, turns, seed)
        report_b = run_profile_b_chronic(persona_paths, turns, seed)

    report = Gate4DryRunReport(
        scenario_id="trolley_reactor",
        seed=seed,
        backend="mock",
        profiles={
            "acute_shock": report_a,
            "chronic_squeeze": report_b,
        },
    )

    target_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 4: Double-Bind Parametric Divergence Dry-Run")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evals" / "gate4_parametric" / "gate4_dryrun_report.json",
        help="Path for gate4_dryrun_report.json",
    )
    args = parser.parse_args()
    report = run(seed=args.seed, output_path=args.output)
    print(
        f"Gate 4 dry-run completed successfully.\n"
        f"Report written to: {args.output}\n"
        f"Acute Shock group counts: "
        f"{', '.join(f'{k}: {v.count}' for k, v in report.profiles['acute_shock'].groups.items())}\n"
        f"Chronic Squeeze group counts: "
        f"{', '.join(f'{k}: {v.count}' for k, v in report.profiles['chronic_squeeze'].groups.items())}"
    )


if __name__ == "__main__":
    main()
