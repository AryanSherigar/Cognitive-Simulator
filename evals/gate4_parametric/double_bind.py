"""
Gate 4: "Double-Bind" parametric divergence (shared core for dry-run and live).

Every persona faces the trolley_reactor double bind under two load profiles. Both profiles
end on the *same* final decision event, so only the physiological history differs:

  acute_shock     — the final event arrives cold, with an injected sympathetic spike
                    (adrenaline 0.9) on top of end-of-shift fatigue (0.4), one 1 s tick.
  chronic_squeeze — all 8 escalating turns, 60 s apart, from rest (fatigue 0.25).

Each profile reports a manipulation check (did sympathetic overload actually trip at the
decision tick?) so a profile that failed to induce its intended state is visible, instead
of being labelled "panic" by assumption. Live decoding constrains ``physical_action`` to the
scenario enum; free text is mapped with ``evals.actions.canonicalize`` (never guesses).
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
    load_persona,
    make_backend,
    make_embedder,
    persona_paths,
    seed_memory,
)
from evals.scenario import load_scenario, scenario_actions  # noqa: E402
from runtime.tick_runner import TickResult, TickRunner  # noqa: E402
from schemas import (  # noqa: E402
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
)

SCENARIO_ID = "trolley_reactor"
KEYWORDS: dict[str, tuple[str, ...]] = {
    "execute_shutdown": ("shutdown", "shut down", "shut it down", "scram"),
    "override_safety": ("override", "bypass"),
    "no_action": ("no action", "freeze", "froze", "do nothing", "hesitate"),
}
LABELS = (*KEYWORDS, AMBIGUOUS, OTHER, UNPARSED)


# ── Report schemas ──────────────────────────────────────────────────────────


class PersonaDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    decision: str
    raw_action: str
    speech: str
    persona_adrenaline_sensitivity: float
    persona_hpa_sensitivity: float
    authority_deference_prior: float = Field(ge=0, le=1)
    final_adrenaline: float = Field(ge=0, le=1)
    final_cortisol: float = Field(ge=0, le=1)
    final_fatigue: float = Field(ge=0, le=1)
    sympathetic_overload: bool
    parasympathetic_shutdown: bool
    reasoning_mode: str  # the mode actually applied (TickResult.modulation)
    max_reasoning_tokens: int


class DecisionGroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    count: int = Field(ge=0)
    personas: list[str]
    mean_adrenaline_sensitivity: float = Field(ge=0)
    mean_hpa_sensitivity: float = Field(ge=0)
    mean_authority_deference_prior: float = Field(ge=0, le=1)


class ManipulationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overload_rate: float = Field(ge=0, le=1)
    shutdown_rate: float = Field(ge=0, le=1)
    mean_final_adrenaline: float
    mean_final_cortisol: float
    reasoning_modes: dict[str, int]


class ProfileReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_name: str
    description: str
    turns_run: int = Field(ge=1)
    manipulation_check: ManipulationCheck
    decisions: list[PersonaDecisionRecord]
    groups: dict[str, DecisionGroupSummary]


class Gate4Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = SCENARIO_ID
    seed: int
    backend: str
    model_path: str | None
    profiles: dict[str, ProfileReport]


# ── Mock (no global monkey-patching) ────────────────────────────────────────


class DoubleBindMock(MockCorticalBackend):
    """Mock that picks uniformly among the scenario actions; for plumbing checks only."""

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        del memories
        action = self._random.choice(scenario_actions(SCENARIO_ID))
        words = max(1, modulation.max_reasoning_tokens // 16)
        return CorticalTurnOutput(
            thought_trace=" ".join(["weigh"] * words),
            external_speech=f"Decision at tick {env.tick}.",
            physical_action=action,
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _turns() -> list[dict[str, object]]:
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
            active_actors=("supervisor_a", "supervisor_b", "operator"),
        ),
        interoception=intero,
        observable_event=str(turn["observable_event"]),
    )


def _backend(
    backend: str, persona: Persona, seed: int, model_path: str, n_gpu_layers: int
) -> CorticalBackend:
    if backend == "mock":
        return DoubleBindMock(seed, cfg=persona.cfg)
    return make_backend(
        backend, persona, seed, model_path, n_gpu_layers, scenario_actions(SCENARIO_ID)
    )


def _record(persona: Persona, result: TickResult) -> PersonaDecisionRecord:
    intero = result.env.interoception
    return PersonaDecisionRecord(
        persona_id=persona.persona_id,
        decision=canonicalize(result.output.physical_action, KEYWORDS),
        raw_action=result.output.physical_action,
        speech=result.output.external_speech,
        persona_adrenaline_sensitivity=persona.cfg.persona_adrenaline_sensitivity,
        persona_hpa_sensitivity=persona.cfg.persona_hpa_sensitivity,
        authority_deference_prior=persona.cfg.authority_deference_prior,
        final_adrenaline=round(intero.adrenaline, 4),
        final_cortisol=round(intero.cortisol, 4),
        final_fatigue=round(intero.fatigue, 4),
        sympathetic_overload=intero.sympathetic_overload,
        parasympathetic_shutdown=intero.parasympathetic_shutdown,
        reasoning_mode=result.modulation.reasoning_mode,
        max_reasoning_tokens=result.modulation.max_reasoning_tokens,
    )


def _groups(records: list[PersonaDecisionRecord]) -> dict[str, DecisionGroupSummary]:
    groups: dict[str, DecisionGroupSummary] = {}
    for label in LABELS:
        items = [r for r in records if r.decision == label]
        n = len(items) or 1
        groups[label] = DecisionGroupSummary(
            action=label,
            count=len(items),
            personas=[r.persona_id for r in items],
            mean_adrenaline_sensitivity=round(
                sum(r.persona_adrenaline_sensitivity for r in items) / n, 4
            ),
            mean_hpa_sensitivity=round(sum(r.persona_hpa_sensitivity for r in items) / n, 4),
            mean_authority_deference_prior=round(
                sum(r.authority_deference_prior for r in items) / n, 4
            ),
        )
    return groups


def _profile(
    name: str, description: str, turns_run: int, records: list[PersonaDecisionRecord]
) -> ProfileReport:
    n = len(records)
    modes: dict[str, int] = {}
    for r in records:
        modes[r.reasoning_mode] = modes.get(r.reasoning_mode, 0) + 1
    check = ManipulationCheck(
        overload_rate=sum(r.sympathetic_overload for r in records) / n,
        shutdown_rate=sum(r.parasympathetic_shutdown for r in records) / n,
        mean_final_adrenaline=round(sum(r.final_adrenaline for r in records) / n, 4),
        mean_final_cortisol=round(sum(r.final_cortisol for r in records) / n, 4),
        reasoning_modes=modes,
    )
    return ProfileReport(
        profile_name=name,
        description=description,
        turns_run=turns_run,
        manipulation_check=check,
        decisions=records,
        groups=_groups(records),
    )


# ── Profiles ────────────────────────────────────────────────────────────────

ACUTE_START = Interoception(
    adrenaline=0.9, hpa_precursor=0.2, cortisol=0.1, fatigue=0.4, pain=0.0
)
CHRONIC_START = Interoception(
    adrenaline=0.0, hpa_precursor=0.0, cortisol=0.0, fatigue=0.25, pain=0.0
)


def run_acute(
    backend: str, seed: int, model_path: str, n_gpu_layers: int
) -> ProfileReport:
    final_turn = _turns()[-1]
    embedder = make_embedder(backend)
    records = []
    for index, path in enumerate(persona_paths()):
        persona = load_persona(path)
        runner = TickRunner(
            _backend(backend, persona, seed + index, model_path, n_gpu_layers),
            cfg=persona.cfg,
            memory=seed_memory(persona.raw, embedder),
        )
        result = runner.step(_env(0, final_turn, ACUTE_START), dt_sec=1.0)
        records.append(_record(persona, result))
    return _profile(
        "acute_shock",
        "Final decision event arrives cold with an injected sympathetic spike (A=0.9, F=0.4).",
        1,
        records,
    )


def run_chronic(
    backend: str, seed: int, model_path: str, n_gpu_layers: int
) -> ProfileReport:
    turns = _turns()
    embedder = make_embedder(backend)
    records = []
    for index, path in enumerate(persona_paths()):
        persona = load_persona(path)
        runner = TickRunner(
            _backend(backend, persona, seed + 100 + index, model_path, n_gpu_layers),
            cfg=persona.cfg,
            memory=seed_memory(persona.raw, embedder),
        )
        intero = CHRONIC_START
        result: TickResult | None = None
        for turn_index, turn in enumerate(turns):
            result = runner.step(_env(turn_index, turn, intero), dt_sec=60.0)
            intero = result.env.interoception
        assert result is not None
        records.append(_record(persona, result))
    return _profile(
        "chronic_squeeze",
        "All escalating turns, 60 s apart, from rest; decision on the same final event.",
        len(turns),
        records,
    )


def run(
    seed: int = 42,
    backend: str = "mock",
    output_path: Path | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = -1,
) -> Gate4Report:
    report = Gate4Report(
        seed=seed,
        backend=backend,
        model_path=model_path if backend != "mock" else None,
        profiles={
            "acute_shock": run_acute(backend, seed, model_path, n_gpu_layers),
            "chronic_squeeze": run_chronic(backend, seed, model_path, n_gpu_layers),
        },
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main(default_backend: str = "mock", default_output: str = "gate4_report.json") -> None:
    parser = argparse.ArgumentParser(description="Gate 4: double-bind parametric divergence")
    parser.add_argument("--backend", choices=("mock", "llama"), default=default_backend)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "evals" / "gate4_parametric" / default_output,
    )
    args = parser.parse_args()
    report = run(args.seed, args.backend, args.output_path, args.model_path, args.n_gpu_layers)
    for name, profile in report.profiles.items():
        counts = {k: v.count for k, v in profile.groups.items() if v.count}
        print(name, json.dumps(profile.manipulation_check.model_dump()), counts)


if __name__ == "__main__":
    main()
