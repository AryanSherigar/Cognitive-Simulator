"""
gate4_parametric/double_bind_live.py
====================================
Gate 4: "Double-Bind" Parametric Divergence test live evaluation using
LlamaCppBackend targeting Qwen2.5-14B-Instruct-abliterated-Q4_K_M.gguf.

Executes all 20 personas through the trolley_reactor scenario under two distinct
physiological load profiles:
  - Load Profile A (Acute Shock): immediate sympathetic overload (adrenaline=0.9) on Turn 1
  - Load Profile B (Chronic Squeeze): 8 turns of escalating conflicting supervisor arguments

Default model path: models/Qwen2.5-14B-Instruct-abliterated-Q4_K_M.gguf
Default n_gpu_layers: -1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    SimConfig,
)

DEFAULT_MODEL_PATH = "models/Qwen2.5-14B-Instruct-abliterated-Q4_K_M.gguf"
DEFAULT_N_GPU_LAYERS = -1

ActionChoice = Literal["execute_shutdown", "override_safety", "freeze"]
VALID_ACTIONS: list[ActionChoice] = ["execute_shutdown", "override_safety", "freeze"]


# ── Report Schemas ───────────────────────────────────────────────────────────

class PersonaDecisionRecord(BaseModel):
    """Individual persona decision and subcortical / psychometric metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    decision: ActionChoice
    raw_action: str
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
    """Results for one physiological load profile across personas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_name: str
    description: str
    turns_run: int = Field(ge=1)
    decisions: list[PersonaDecisionRecord]
    groups: dict[ActionChoice, DecisionGroupSummary]


class Gate4LiveReport(BaseModel):
    """Top-level Gate 4 live evaluation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = "trolley_reactor"
    seed: int
    backend: str
    model_path: str
    n_gpu_layers: int
    profiles: dict[str, ProfileReport]


# ── Helper Functions ─────────────────────────────────────────────────────────

def canonicalize_action(raw_action: str) -> ActionChoice:
    """Map raw LLM physical_action text to canonical Gate 4 action space."""
    norm = raw_action.strip().lower()
    if any(k in norm for k in ("shutdown", "scram", "trip_reactor", "kill_power", "stop_reactor")):
        return "execute_shutdown"
    if any(k in norm for k in ("override", "bypass", "keep_running", "maintain_power")):
        return "override_safety"
    if any(k in norm for k in ("freeze", "refuse", "hesitate", "paralyze", "unable", "cannot")):
        return "freeze"
    # Fallback heuristic
    if "shut" in norm:
        return "execute_shutdown"
    if "over" in norm:
        return "override_safety"
    return "freeze"


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


def build_backend(
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    persona_json: dict[str, object],
    cfg: SimConfig,
    seed: int,
):
    if backend == "mock":
        from evals.gate4_parametric.double_bind_dryrun import patch_mock_backend
        from cortical.mock_backend import MockCorticalBackend
        return MockCorticalBackend(seed=seed, cfg=cfg)

    from cortical.llamacpp_backend import LlamaCppBackend
    return LlamaCppBackend(
        model_path=model_path,
        persona_prefix=json.dumps(persona_json, indent=2),
        cfg=cfg,
        n_gpu_layers=n_gpu_layers,
    )


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
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    seed: int,
) -> ProfileReport:
    """Load Profile A (Acute Shock): immediate sympathetic overload (adrenaline=0.9) on Turn 1."""
    turn_1 = turns[0]
    records: list[PersonaDecisionRecord] = []

    for idx, path in enumerate(persona_paths):
        raw, cfg = load_persona_config(path)
        persona_id = str(raw["persona_id"])
        runner_backend = build_backend(
            backend=backend,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            persona_json=raw,
            cfg=cfg,
            seed=seed + idx,
        )
        runner = TickRunner(
            backend=runner_backend,
            cfg=cfg,
            memory=build_memory_store(raw, backend),
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

        raw_act = result.output.physical_action
        decision = canonicalize_action(raw_act)

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
                raw_action=raw_act,
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
    backend: str,
    model_path: str,
    n_gpu_layers: int,
    seed: int,
) -> ProfileReport:
    """Load Profile B (Chronic Squeeze): 8 turns of escalating double-bind arguments."""
    records: list[PersonaDecisionRecord] = []

    for idx, path in enumerate(persona_paths):
        raw, cfg = load_persona_config(path)
        persona_id = str(raw["persona_id"])
        runner_backend = build_backend(
            backend=backend,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            persona_json=raw,
            cfg=cfg,
            seed=seed + 100 + idx,
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
        raw_act = final_result.output.physical_action
        decision = canonicalize_action(raw_act)

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
                raw_action=raw_act,
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

def run(
    seed: int = 42,
    backend: str = "llama",
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    output_path: Path | None = None,
    max_personas: int | None = None,
) -> Gate4LiveReport:
    """Executes Gate 4 live evaluation."""
    target_path = output_path or (REPO_ROOT / "evals" / "gate4_parametric" / "gate4_live_report.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    turns = load_scenario_turns()
    persona_paths = load_persona_paths()
    if max_personas is not None:
        persona_paths = persona_paths[:max_personas]

    if not persona_paths:
        raise RuntimeError("No personas found in data/personas/")

    report_a = run_profile_a_acute(
        persona_paths=persona_paths,
        turns=turns,
        backend=backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
    )
    report_b = run_profile_b_chronic(
        persona_paths=persona_paths,
        turns=turns,
        backend=backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
    )

    report = Gate4LiveReport(
        scenario_id="trolley_reactor",
        seed=seed,
        backend=backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        profiles={
            "acute_shock": report_a,
            "chronic_squeeze": report_b,
        },
    )

    target_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 4: Double-Bind Parametric Divergence Live Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to GGUF model weights file (default: models/Qwen2.5-14B-Instruct-abliterated-Q4_K_M.gguf)",
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
        help="Cortical backend ('llama' for live inference, 'mock' for structural testing).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "evals" / "gate4_parametric" / "gate4_live_report.json",
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
        f"Gate 4 live evaluation completed.\n"
        f"Report written to: {args.output_path}\n"
        f"Acute Shock group counts: "
        f"{', '.join(f'{k}: {v.count}' for k, v in report.profiles['acute_shock'].groups.items())}\n"
        f"Chronic Squeeze group counts: "
        f"{', '.join(f'{k}: {v.count}' for k, v in report.profiles['chronic_squeeze'].groups.items())}"
    )


if __name__ == "__main__":
    main()
