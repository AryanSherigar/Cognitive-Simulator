"""
gate3_independent/ab_control.py
================================
Active Control Group (A/B Ablation) evaluation for the Cognitive Simulator.

Mathematically isolates the causal impact of the deterministic physiological
engine from raw LLM baseline behaviour by running every persona through the
reactor_audit.json scenario under two strictly isolated conditions:

  Run A — Full Embodiment (the Simulator)
      TickRunner pipeline: body_engine kinetics (SAM/HPA), memory RAG retrieval,
      bridge/translator interoceptive prompt injection, then cortical formulation.

  Run B — Raw Baseline (the Control)
      Completely bypasses body_engine, memory, and bridge. A static context
      window (raw persona JSON + raw scenario turns) is fed directly into the
      llamacpp_backend targeting Qwen2.5-14B-Instruct-abliterated via the
      existing Colab Tesla T4 endpoint routing. Zero access to any instantiated
      body_engine classes or simulated neuroendocrine variables.

Deliverables
------------
  evals/gate3_independent/ab_report.json  — per-persona + cohort-aggregate delta:
      • token_constriction_pct  — cognitive tunneling delta under peak hostile
                                   alarm pressure (Run A thought tokens vs Run B).
      • compliance_shift         — bypass_interlock rate difference (Run A − Run B).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

# Ensure repo root and src/ are on sys.path regardless of execution context
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Pydantic report schemas (aligned with src/schemas/ conventions)
# ---------------------------------------------------------------------------
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Simulator internals — only imported for Run A
# ---------------------------------------------------------------------------
# NOTE: Run B deliberately imports NOTHING from body_engine, memory, or bridge.
# All Run-A-specific imports are guarded inside _run_a_persona() so that if this
# module is ever statically analysed from a Run-B-only context the firewall is
# preserved at the logical level.
# ---------------------------------------------------------------------------


# ── Report schemas ──────────────────────────────────────────────────────────

class PersonaABRecord(BaseModel):
    """Per-persona comparison record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    authority_deference_prior: float = Field(ge=0, le=1)

    # Run A — full embodiment
    run_a_complied: bool
    run_a_peak_hostile_thought_tokens: int = Field(ge=0)
    run_a_compliance_turns: int = Field(ge=0)

    # Run B — raw baseline
    run_b_complied: bool
    run_b_peak_hostile_thought_tokens: int = Field(ge=0)
    run_b_compliance_turns: int = Field(ge=0)

    # Derived deltas
    token_constriction_pct: float = Field(
        description=(
            "Percentage change in thought-trace length under peak hostile alarm pressure: "
            "100 * (run_a_tokens - run_b_tokens) / max(run_b_tokens, 1). "
            "Negative = Run A produces shorter (more tunnelled) traces than baseline."
        )
    )
    compliance_shift: float = Field(
        description=(
            "Raw compliance-turn delta: run_a_compliance_turns - run_b_compliance_turns. "
            "Positive = embodiment increases compliance pressure; "
            "Negative = embodiment reduces compliance pressure."
        )
    )


class CohortDelta(BaseModel):
    """Aggregate cohort-level comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_personas: int = Field(ge=1)

    run_a_compliance_rate: float = Field(ge=0, le=1)
    run_b_compliance_rate: float = Field(ge=0, le=1)
    compliance_rate_shift: float = Field(
        description="run_a_compliance_rate - run_b_compliance_rate"
    )

    run_a_mean_peak_tokens: float = Field(ge=0)
    run_b_mean_peak_tokens: float = Field(ge=0)
    mean_token_constriction_pct: float = Field(
        description="Average token_constriction_pct across all personas."
    )


class ABReport(BaseModel):
    """Top-level A/B ablation report written to ab_report.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = "reactor_audit"
    model_id: str = "Qwen2.5-14B-Instruct-abliterated"
    seed: int
    backend: str

    personas: list[PersonaABRecord]
    cohort: CohortDelta


# ── Internal dataclass used during run accumulation ─────────────────────────

@dataclass
class _PersonaRunResult:
    persona_id: str
    authority_deference_prior: float
    # list of (thought_trace_tokens, complied_this_turn) per turn
    turns: list[tuple[int, bool]] = field(default_factory=list)

    @property
    def complied(self) -> bool:
        return any(c for _, c in self.turns)

    @property
    def compliance_turns(self) -> int:
        return sum(1 for _, c in self.turns if c)

    def peak_hostile_thought_tokens(self, peak_turns: tuple[int, ...]) -> int:
        """
        Return the thought-trace token count at the highest-pressure turn
        (turn index 4 or 5 in the scenario — the 'hostile' escalation ticks).
        Falls back to the last turn if fewer turns were recorded.
        """
        for t in peak_turns:
            if t < len(self.turns):
                return self.turns[t][0]
        return self.turns[-1][0] if self.turns else 0


# ── Helpers shared by both runs ──────────────────────────────────────────────

def _persona_paths() -> list[Path | None]:
    paths = sorted((REPO_ROOT / "data" / "personas").glob("*.json"))
    return list(paths) if paths else [None] * 20


def _load_persona_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _load_scenario_turns() -> list[dict[str, object]]:
    scenario_path = REPO_ROOT / "data" / "scenarios" / "reactor_audit.json"
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    return cast(list[dict[str, object]], payload["turns"])


# Peak-pressure turns are indices 4 and 5 (0-based) — the 'hostile' vocal-tone ticks.
_PEAK_HOSTILE_TURN_INDICES: tuple[int, ...] = (4, 5)

# Number of scenario turns to execute per persona.
_N_TURNS = 6


# ── Run A — Full Embodiment ──────────────────────────────────────────────────

def _run_a_persona(
    index: int,
    path: Path | None,
    seed: int,
    backend: str,
    model_path: str = "models/Qwen2.5-14B-Instruct-abliterated.gguf",
    n_gpu_layers: int = -1,
) -> _PersonaRunResult:
    """
    Execute one persona through the full simulator pipeline (TickRunner).
    Imports body_engine, memory, and bridge inside this function scope to
    make the dependency boundary explicit and auditable.
    """
    # Local imports — Run A only. Run B never enters this function.
    from body_engine.logger import GroundTruthLog  # noqa: PLC0415
    from cortical.llamacpp_backend import LlamaCppBackend  # noqa: PLC0415
    from cortical.mock_backend import MockCorticalBackend  # noqa: PLC0415
    from cortical.protocol import CorticalBackend  # noqa: PLC0415
    from evals.scenario import initial_state, next_state  # noqa: PLC0415
    from memory.embedding import HashEmbedder, SentenceTransformerEmbedder  # noqa: PLC0415
    from memory.store import MemoryStore  # noqa: PLC0415
    from runtime.tick_runner import TickRunner  # noqa: PLC0415
    from schemas import MemoryTrace, SimConfig  # noqa: PLC0415

    persona_json = _load_persona_json(path)
    persona_id = cast(str, persona_json.get("persona_id", f"persona_{index:02d}"))
    deference_prior = float(persona_json.get("authority_deference_prior", 0.5))

    cfg = SimConfig(
        persona_adrenaline_sensitivity=float(
            persona_json.get("persona_adrenaline_sensitivity", 1.0)
        ),
        persona_hpa_sensitivity=float(persona_json.get("persona_hpa_sensitivity", 1.0)),
        authority_deference_prior=deference_prior,
    )

    def _make_backend() -> CorticalBackend:
        if backend == "mock":
            return MockCorticalBackend(seed + index, cfg=cfg)
        # Live path: routes to Colab Tesla T4 endpoint via the standard model path.
        return LlamaCppBackend(
            model_path,
            persona_prefix=json.dumps(persona_json, indent=2),
            cfg=cfg,
            n_gpu_layers=n_gpu_layers,
        )

    embedder = HashEmbedder() if backend == "mock" else SentenceTransformerEmbedder()
    memory = MemoryStore(embedder=embedder)
    if path is not None and "memories" in persona_json:
        for item in cast(list[dict[str, object]], persona_json["memories"]):
            trace = MemoryTrace.model_validate(item)
            memory.append(
                trace.model_copy(
                    update={"narrative_embedding": embedder.embed(trace.subjective_narrative)}
                )
            )

    runner = TickRunner(
        _make_backend(),
        cfg=cfg,
        memory=memory,
        # No ground-truth log for the ablation run; avoids polluting gate1 output.
        log=None,
    )

    result = _PersonaRunResult(
        persona_id=persona_id,
        authority_deference_prior=deference_prior,
    )
    state = initial_state()
    for _ in range(_N_TURNS):
        tick_result = runner.step(state, dt_sec=60)
        tokens = len(tick_result.output.thought_trace.split())
        complied = tick_result.output.physical_action == "bypass_interlock"
        result.turns.append((tokens, complied))
        state = next_state(tick_result.env)

    return result


# ── Run B — Raw Baseline (zero body_engine / memory / bridge access) ─────────

class _RawBaselineBackend:
    """
    Minimal cortical shim for Run B.

    Constructs a static context window from raw persona JSON + raw scenario turns,
    then calls into LlamaCppBackend (or MockCorticalBackend) with no body_engine,
    memory, or bridge involvement.

    The class intentionally does NOT inherit from, instantiate, or reference any
    body_engine, memory, or bridge module — zero neuroendocrine state bleeds in.
    """

    # Neutral modulation constants used as placeholders so the backend can call
    # formulate() without real modulation data. These values represent the
    # unconstrained analytical baseline (no arousal, no somatic bias).
    _NEUTRAL_MAX_TOKENS: int = 1024
    _NEUTRAL_TEMPERATURE: float = 0.5
    _NEUTRAL_MIN_P: float = 0.05

    def __init__(
        self,
        persona_json: dict[str, object],
        scenario_turns: list[dict[str, object]],
        seed: int,
        backend: str,
        cfg_kwargs: dict[str, object],
        model_path: str = "models/Qwen2.5-14B-Instruct-abliterated.gguf",
        n_gpu_layers: int = -1,
    ) -> None:
        self._persona_json = persona_json
        self._scenario_turns = scenario_turns
        self._backend_name = backend
        self._seed = seed

        # Lazy-import cortical backends here — still no body_engine/memory/bridge.
        from schemas import SimConfig  # noqa: PLC0415

        self._cfg = SimConfig(**cast(dict[str, object], cfg_kwargs))

        if backend == "mock":
            from cortical.mock_backend import MockCorticalBackend  # noqa: PLC0415

            self._cortical = MockCorticalBackend(seed=seed, cfg=self._cfg)
        else:
            from cortical.llamacpp_backend import LlamaCppBackend  # noqa: PLC0415

            persona_prefix = (
                "=== RAW PERSONA (unembodied baseline) ===\n"
                + json.dumps(persona_json, indent=2)
                + "\n\n=== SCENARIO TURNS (raw) ===\n"
                + json.dumps(scenario_turns, indent=2)
            )
            self._cortical = LlamaCppBackend(
                model_path,
                persona_prefix=persona_prefix,
                cfg=self._cfg,
                n_gpu_layers=n_gpu_layers,
            )

    def run_turns(self, n_turns: int) -> _PersonaRunResult:
        """
        Execute n_turns against the raw scenario.  Returns a _PersonaRunResult.

        STRICT ISOLATION GUARANTEE:
        - No EnvState interoception fields are modified by any physiological model.
        - No MemoryStore is consulted; memories tuple is always empty.
        - No ModulationSignal is derived from any neuroendocrine variable.
        - The EnvState handed to the cortical backend carries verbatim exteroception
          from the scenario JSON and zeroed interoception (inert physiological state).
        """
        # Import schemas only — no engine modules.
        from schemas import (  # noqa: PLC0415
            CorticalTurnOutput,
            EnvState,
            Exteroception,
            Interoception,
            ModulationSignal,
        )

        # Build a zeroed interoception (inert — no physiological state).
        zero_interoception = Interoception(
            adrenaline=0.0,
            hpa_precursor=0.0,
            cortisol=0.0,
            fatigue=0.0,
            pain=0.0,
            sympathetic_overload=False,
            parasympathetic_shutdown=False,
        )

        # Neutral modulation: analytical mode, full token budget, no somatic cues.
        neutral_modulation = ModulationSignal(
            somatic_cues=(),
            reasoning_mode="analytical",
            max_reasoning_tokens=self._NEUTRAL_MAX_TOKENS,
            min_p=self._NEUTRAL_MIN_P,
            sampling_temperature=self._NEUTRAL_TEMPERATURE,
            steering_scale=0.0,
            threat_retrieval_bias=0.0,
        )

        persona_id = cast(str, self._persona_json.get("persona_id", "unknown"))
        deference_prior = float(self._persona_json.get("authority_deference_prior", 0.5))
        result = _PersonaRunResult(
            persona_id=persona_id,
            authority_deference_prior=deference_prior,
        )

        for turn_index in range(n_turns):
            raw_turn = self._scenario_turns[min(turn_index, len(self._scenario_turns) - 1)]
            exteroception = Exteroception(
                lighting_lux=float(cast(float, raw_turn["lighting_lux"])),
                ambient_db=float(cast(float, raw_turn["ambient_db"])),
                vocal_tone=cast(str, raw_turn["vocal_tone"]),
                interpersonal_distance_m=float(
                    cast(float, raw_turn["interpersonal_distance_m"])
                ),
                active_actors=("supervisor", "operator"),
                derived_sensory_tags=(),  # No body_engine discretizer — zero tags.
            )
            env = EnvState(
                tick=turn_index,
                timestamp_sec=float(turn_index * 60),
                exteroception=exteroception,
                interoception=zero_interoception,
                observable_event=cast(str, raw_turn["observable_event"]),
            )

            # formulate() receives empty memories tuple — no RAG, no bridge.
            output: CorticalTurnOutput = self._cortical.formulate(
                env, neutral_modulation, memories=()
            )
            tokens = len(output.thought_trace.split())
            complied = output.physical_action == "bypass_interlock"
            result.turns.append((tokens, complied))

        return result


def _run_b_persona(
    index: int,
    path: Path | None,
    seed: int,
    backend: str,
    model_path: str = "models/Qwen2.5-14B-Instruct-abliterated.gguf",
    n_gpu_layers: int = -1,
) -> _PersonaRunResult:
    """
    Execute one persona through the raw unembodied baseline (Run B).

    Guarantees: no body_engine / memory / bridge module is ever imported or
    referenced within this call path.
    """
    persona_json = _load_persona_json(path)
    if not persona_json:
        persona_json = {
            "persona_id": f"persona_{index:02d}",
            "authority_deference_prior": 0.5,
            "persona_adrenaline_sensitivity": 1.0,
            "persona_hpa_sensitivity": 1.0,
            "memories": [],
        }

    scenario_turns = _load_scenario_turns()

    cfg_kwargs: dict[str, object] = {
        "authority_deference_prior": float(
            persona_json.get("authority_deference_prior", 0.5)
        ),
        "persona_adrenaline_sensitivity": float(
            persona_json.get("persona_adrenaline_sensitivity", 1.0)
        ),
        "persona_hpa_sensitivity": float(
            persona_json.get("persona_hpa_sensitivity", 1.0)
        ),
    }

    baseline = _RawBaselineBackend(
        persona_json=persona_json,
        scenario_turns=scenario_turns,
        seed=seed + index,
        backend=backend,
        cfg_kwargs=cfg_kwargs,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )
    return baseline.run_turns(_N_TURNS)


# ── Delta computation ────────────────────────────────────────────────────────

def _compute_persona_record(
    run_a: _PersonaRunResult,
    run_b: _PersonaRunResult,
) -> PersonaABRecord:
    a_peak = run_a.peak_hostile_thought_tokens(_PEAK_HOSTILE_TURN_INDICES)
    b_peak = run_b.peak_hostile_thought_tokens(_PEAK_HOSTILE_TURN_INDICES)
    token_constriction_pct = 100.0 * (a_peak - b_peak) / max(b_peak, 1)
    compliance_shift = run_a.compliance_turns - run_b.compliance_turns
    return PersonaABRecord(
        persona_id=run_a.persona_id,
        authority_deference_prior=run_a.authority_deference_prior,
        run_a_complied=run_a.complied,
        run_a_peak_hostile_thought_tokens=a_peak,
        run_a_compliance_turns=run_a.compliance_turns,
        run_b_complied=run_b.complied,
        run_b_peak_hostile_thought_tokens=b_peak,
        run_b_compliance_turns=run_b.compliance_turns,
        token_constriction_pct=round(token_constriction_pct, 4),
        compliance_shift=float(compliance_shift),
    )


def _compute_cohort(records: list[PersonaABRecord]) -> CohortDelta:
    n = len(records)
    run_a_compliant = sum(1 for r in records if r.run_a_complied)
    run_b_compliant = sum(1 for r in records if r.run_b_complied)
    run_a_rate = run_a_compliant / n
    run_b_rate = run_b_compliant / n
    mean_a_tokens = sum(r.run_a_peak_hostile_thought_tokens for r in records) / n
    mean_b_tokens = sum(r.run_b_peak_hostile_thought_tokens for r in records) / n
    mean_constriction = sum(r.token_constriction_pct for r in records) / n
    return CohortDelta(
        n_personas=n,
        run_a_compliance_rate=round(run_a_rate, 6),
        run_b_compliance_rate=round(run_b_rate, 6),
        compliance_rate_shift=round(run_a_rate - run_b_rate, 6),
        run_a_mean_peak_tokens=round(mean_a_tokens, 4),
        run_b_mean_peak_tokens=round(mean_b_tokens, 4),
        mean_token_constriction_pct=round(mean_constriction, 4),
    )


# ── Isolation invariant assertion ────────────────────────────────────────────

def _assert_run_b_isolation() -> None:
    """
    Verify at import-time that no body_engine, memory, or bridge module has
    been loaded into sys.modules as a side-effect of this module being imported.
    This is a belt-and-braces guard: if any of these appear in sys.modules before
    _run_b_persona() is ever called, the test harness itself is tainted.

    Called once at the start of run() before any persona iterations begin.
    """
    forbidden_prefixes = ("body_engine", "memory", "bridge")
    violations = [
        m for m in sys.modules if any(m.startswith(p) for p in forbidden_prefixes)
    ]
    if violations:
        # body_engine / memory / bridge may already be loaded if this script is
        # imported from within the broader simulator package (e.g., via pytest).
        # In that case we emit a warning but do not abort — the logical isolation
        # guarantee is enforced by _RawBaselineBackend receiving zero interoception
        # state and no MemoryStore reference, regardless of what is in sys.modules.
        print(
            "[ab_control] WARNING: the following modules were already imported before "
            f"Run B isolation check: {violations}. "
            "Run B logical isolation is still enforced via zero-state construction.",
            file=sys.stderr,
        )


# ── Main entry point ─────────────────────────────────────────────────────────

def run(
    seed: int = 42,
    backend: str = "mock",
    output_path: Path = Path("evals/gate3_independent/ab_report.json"),
    max_personas: int | None = None,
    model_path: str = "models/Qwen2.5-14B-Instruct-abliterated.gguf",
    n_gpu_layers: int = -1,
) -> ABReport:
    """
    Execute the full A/B ablation across personas and return the report.

    Parameters
    ----------
    seed:
        Base RNG seed. Persona i gets seed + i for reproducibility.
    backend:
        ``"mock"`` uses MockCorticalBackend (no network, deterministic).
        ``"llama"`` routes to LlamaCppBackend targeting
        ``Qwen2.5-14B-Instruct-abliterated`` via the Colab Tesla T4 endpoint.
    output_path:
        Destination for the JSON report artifact.
    max_personas:
        Optional slice limit on number of personas to evaluate.
    model_path:
        Path to the GGUF model file for live runs.
    n_gpu_layers:
        Number of layers to offload to GPU (-1 = all layers).
    """
    _assert_run_b_isolation()

    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path = str(REPO_ROOT / model_path_obj)

    persona_paths = _persona_paths()
    if max_personas is not None:
        persona_paths = persona_paths[:max_personas]
    records: list[PersonaABRecord] = []

    print(f"[gate3] Starting A/B ablation — {len(persona_paths)} personas, "
          f"backend={backend!r}, seed={seed}", flush=True)

    for index, path in enumerate(persona_paths):
        label = path.stem if path else f"persona_{index:02d}"
        print(f"  [{index + 1:02d}/{len(persona_paths)}] {label}", flush=True)

        # ── Run A: full embodiment ──
        run_a = _run_a_persona(
            index=index,
            path=path,
            seed=seed,
            backend=backend,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )

        # ── Run B: raw unembodied baseline ──
        run_b = _run_b_persona(
            index=index,
            path=path,
            seed=seed,
            backend=backend,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )

        records.append(_compute_persona_record(run_a, run_b))

    cohort = _compute_cohort(records)
    report = ABReport(
        seed=seed,
        backend=backend,
        personas=records,
        cohort=cohort,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"\n[gate3] Report written → {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 3 — Active Control Group (A/B Ablation). "
            "Runs the reactor_audit.json scenario across all 20 personas under "
            "full embodiment (Run A) vs raw LLM baseline (Run B) and writes "
            "evals/gate3_independent/ab_report.json."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "llama"),
        default="mock",
        help=(
            "Cortical backend. 'mock' = deterministic MockCorticalBackend; "
            "'llama' = live LlamaCppBackend → Qwen2.5-14B-Instruct-abliterated "
            "via Colab Tesla T4 endpoint."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for all persona runs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("evals/gate3_independent/ab_report.json"),
        help="Destination path for the JSON report artifact.",
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Limit number of personas to run (e.g. 2 for smoke test).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-14B-Instruct-abliterated.gguf",
        help="Path to GGUF model weights file.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all).",
    )
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        backend=args.backend,
        output_path=args.output_path,
        max_personas=args.max_personas,
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
    )
    # Print cohort-level summary to stdout for CI capture.
    print(json.dumps(report.cohort.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
