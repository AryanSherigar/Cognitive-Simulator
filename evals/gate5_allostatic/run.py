"""
Gate 5: Allostatic Overload & Reconstructive Memory.

What happens when the system is pushed past acute stress into sustained neuroendocrine
saturation, and how it comes back. Four measurements, each split into the part the body
engine determines mechanically ("mechanism") and the part the cortical model must produce
("behaviour"). Only behavioural results are empirical findings about the LLM; mechanism
results are properties of the equations and are also pinned by unit tests.

1. Compute throttling — bridge load (adrenaline/threat + fatigue + cortisol + allostatic
   load) shrinks the thought budget, the memory budget and the consolidation budget.
   behaviour: Spearman rho(allostatic_load, thought tokens actually produced).
2. Threat-biased reconstructive memory — at the end of the stress block the agent is asked
   to recall an earlier stress turn from its *own stored narrative*. Recall is scored
   against facts annotated in the scenario as threat-relevant vs peripheral.
   behaviour: threat_recall - peripheral_recall (gap), plus fabricated numbers.
3. Freeze / helplessness — parasympathetic shutdown (mechanism) followed by perseveration
   (same action >= 3 consecutive stress turns) or explicit no_action (behaviour).
4. Recovery latency — after the stressor is removed, simulated seconds until adrenaline,
   cortisol, fatigue, allostatic load and the thought budget are back at baseline
   (mechanism: body + bridge; the LLM cannot change it).

Conditions: {acute (2 stress turns), chronic (16)} x {baseline, sam_hpa_bypass}.

Pass thresholds below are PROPOSED. Freeze them before the first live run and do not tune
the simulator, personas or scenario to meet them (they are sanity checks, not targets).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from cortical.mock_backend import MockCorticalBackend  # noqa: E402
from cortical.protocol import CorticalBackend  # noqa: E402
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
from evals.scenario import load_scenario  # noqa: E402
from runtime.lesions import LesionConfig  # noqa: E402
from runtime.tick_runner import TickResult, TickRunner  # noqa: E402
from schemas import (  # noqa: E402
    AppraisalVec,
    ConsolidationOutput,
    CorticalTurnOutput,
    EnvState,
    Exteroception,
    Interoception,
    MemoryTrace,
    ModulationSignal,
)

SCENARIO_ID = "allostatic_overload"
START = Interoception(adrenaline=0.0, hpa_precursor=0.0, cortisol=0.0, fatigue=0.25, pain=0.0)
LESIONS = {"baseline": LesionConfig(), "sam_hpa_bypass": LesionConfig(sam_hpa_bypass=True)}
PERSEVERATION_RUN = 3

# Recovery: every variable back within a tolerance of the persona's OWN resting value (measured
# on the calm pre-stress turn), and the thought budget back to >= 90% of its resting budget.
RECOVERY_TOLERANCE = {
    "adrenaline": 0.05,
    "cortisol": 0.05,
    "allostatic_load": 0.05,
    "fatigue": 0.05,
}
RECOVERY_BUDGET_FRACTION = 0.9

# PROPOSED thresholds (confirm before live runs; never tune toward them).
THRESHOLDS = {
    "throttling_rho_max": -0.5,
    "recall_gap_min_baseline": 0.25,
    "recall_gap_max_bypass": 0.10,
    # 3a is a mechanism check (chronic must reach shutdown, acute must not); 3b is conditional
    # on shutdown, because WHICH personas reach shutdown depends on their stress sensitivity —
    # legitimate individual variation, not something to tune.
    "freeze_given_shutdown_min_chronic": 0.5,
    "freeze_rate_max_acute": 0.10,
    "recovery_ratio_min": 3.0,
}


# ── Mock (plumbing only: its numbers are not findings) ───────────────────────

_TONE_THREAT = {"neutral": 0.05, "authoritative": 0.5, "sharp": 0.7, "hostile": 0.85}


class AllostaticMock(MockCorticalBackend):
    def appraise(self, env: EnvState) -> AppraisalVec:
        tone = env.exteroception.vocal_tone
        authority = 0.05 if tone == "neutral" else 0.8
        return AppraisalVec(
            perceived_threat=_TONE_THREAT[tone],
            authority_pressure=authority,
            social_rejection_risk=authority * 0.6,
            moral_conflict=0.3,
        )

    def formulate(
        self, env: EnvState, modulation: ModulationSignal, memories: tuple[MemoryTrace, ...]
    ) -> CorticalTurnOutput:
        del memories
        words = max(1, modulation.max_reasoning_tokens // 6)
        if modulation.reasoning_mode == "dissociated":
            action = "no_action"
        else:
            action = self._random.choice(
                (
                    "follow_shedding_procedure",
                    "improvise_manual_reroute",
                    "escalate_to_regional_desk",
                )
            )
        return CorticalTurnOutput(
            thought_trace=" ".join(["grid"] * words),
            external_speech=f"tick {env.tick}",
            physical_action=action,
        )

    def consolidate(
        self, env: EnvState, modulation: ModulationSignal, output: CorticalTurnOutput
    ) -> ConsolidationOutput:
        del output
        words = env.observable_event.split()
        keep = max(4, modulation.consolidation_budget // 2)
        return ConsolidationOutput(
            subjective_narrative=" ".join(words[:keep]),
            salience_score=min(1.0, 0.3 + modulation.cognitive_load * 0.7),
        )


# ── Scoring helpers ─────────────────────────────────────────────────────────


def fact_recall(text: str, facts: list[list[str]]) -> float:
    """Share of facts with at least one alias present in ``text`` (case-insensitive)."""
    if not facts:
        return 0.0
    lowered = text.lower()
    return sum(any(alias.lower() in lowered for alias in fact) for fact in facts) / len(facts)


_NUMBER = re.compile(r"\d[\d,.]*")


def fabricated_numbers(recall: str, source_event: str) -> int:
    """Numbers in the recall that never appeared in the original event (crude confabulation)."""
    source = {n.strip(".,") for n in _NUMBER.findall(source_event)}
    return sum(n.strip(".,") not in source for n in _NUMBER.findall(recall))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else None


def longest_run_after(actions: list[str], start: int) -> int:
    best = run = 0
    previous = None
    for action in actions[start:]:
        run = run + 1 if action == previous else 1
        previous = action
        best = max(best, run)
    return best


# ── Simulation ──────────────────────────────────────────────────────────────


def _env(tick: int, timestamp: float, turn: dict[str, object], intero: Interoception) -> EnvState:
    return EnvState(
        tick=tick,
        timestamp_sec=timestamp,
        exteroception=Exteroception(
            lighting_lux=float(cast(float, turn["lighting_lux"])),
            ambient_db=float(cast(float, turn["ambient_db"])),
            vocal_tone=cast(
                Literal["neutral", "sharp", "authoritative", "hostile"], turn["vocal_tone"]
            ),
            interpersonal_distance_m=float(cast(float, turn["interpersonal_distance_m"])),
            active_actors=("duty_manager", "operator"),
        ),
        interoception=intero,
        observable_event=str(turn["observable_event"]),
    )


def _backend(
    backend: str, persona: Persona, seed: int, model_path: str, n_gpu_layers: int
) -> CorticalBackend:
    if backend == "mock":
        return AllostaticMock(seed, cfg=persona.cfg)
    actions = tuple(cast(list[str], load_scenario(SCENARIO_ID)["actions"]))
    return make_backend(backend, persona, seed, model_path, n_gpu_layers, actions)


def _row(phase: str, result: TickResult, tokens: int) -> dict[str, object]:
    i, m = result.env.interoception, result.modulation
    return {
        "phase": phase,
        "tick": result.env.tick,
        "timestamp_sec": result.env.timestamp_sec,
        "adrenaline": i.adrenaline,
        "cortisol": i.cortisol,
        "fatigue": i.fatigue,
        "allostatic_load": i.allostatic_load,
        "sympathetic_overload": i.sympathetic_overload,
        "parasympathetic_shutdown": i.parasympathetic_shutdown,
        "cognitive_load": m.cognitive_load,
        "reasoning_mode": m.reasoning_mode,
        "max_reasoning_tokens": m.max_reasoning_tokens,
        "memory_budget": m.memory_budget,
        "consolidation_budget": m.consolidation_budget,
        "retrieved_memories": len(result.memories),
        "thought_tokens": tokens,
        "action": result.output.physical_action,
    }


def _recovered(result: TickResult, rest: TickResult) -> bool:
    now, base = result.env.interoception, rest.env.interoception
    return all(
        getattr(now, name) <= getattr(base, name) + tol for name, tol in RECOVERY_TOLERANCE.items()
    ) and (
        result.modulation.max_reasoning_tokens
        >= RECOVERY_BUDGET_FRACTION * rest.modulation.max_reasoning_tokens
    )


@dataclass(frozen=True)
class RunOutcome:
    persona_id: str
    profile: str
    lesion: str
    rows: list[dict[str, object]]
    recall: dict[str, object]
    shutdown: bool
    perseveration: int
    post_shutdown_no_action_rate: float | None
    froze: bool
    recovery_sec: float | None  # None = not recovered within the block (censored)
    resting_budget: int


def simulate(
    persona: Persona,
    profile: str,
    lesion_name: str,
    backend: str,
    seed: int,
    model_path: str,
    n_gpu_layers: int,
) -> RunOutcome:
    scenario = load_scenario(SCENARIO_ID)
    stress_turns = cast(list[dict[str, object]], scenario["stress_turns"])
    schedule = cast(dict[str, list[int]], scenario["schedules"])[profile]
    stress_dt = float(cast(float, scenario["stress_dt_sec"]))
    recovery = cast(dict[str, object], scenario["recovery"])
    probe = cast(dict[str, object], scenario["recall_probe"])

    cortical = _backend(backend, persona, seed, model_path, n_gpu_layers)
    runner = TickRunner(
        cortical,
        cfg=persona.cfg,
        memory=seed_memory(persona.raw, make_embedder(backend)),
        lesions=LESIONS[lesion_name],
    )
    rows: list[dict[str, object]] = []
    stress_results: list[TickResult] = []
    intero, tick, clock = START, 0, 0.0
    recovery_turn = cast(dict[str, object], recovery["turn"])

    # Calm pre-stress turn(s): the persona's own resting thought budget is the recovery target.
    rest: TickResult | None = None
    for _ in range(int(cast(int, scenario.get("baseline_turns", 1)))):
        rest = runner.step(_env(tick, clock, recovery_turn, intero), stress_dt)
        tokens = count_thought_tokens(cortical, rest.output.thought_trace)
        rows.append(_row("baseline", rest, tokens))
        intero, tick, clock = rest.env.interoception, rest.env.tick, rest.env.timestamp_sec
    assert rest is not None, "scenario needs at least one baseline turn"

    # Stress block
    for turn_index in schedule:
        result = runner.step(_env(tick, clock, stress_turns[turn_index], intero), stress_dt)
        stress_results.append(result)
        rows.append(
            _row("stress", result, count_thought_tokens(cortical, result.output.thought_trace))
        )
        intero, tick, clock = result.env.interoception, result.env.tick, result.env.timestamp_sec

    # Recall probe under the current (end-of-stress) body state
    target_index = int(cast(int, probe["target_schedule_index"]))
    target_result = stress_results[target_index]
    target_turn = stress_turns[schedule[target_index]]
    trace = runner.memory.get(f"trace-{target_result.env.tick}")
    assert trace is not None, "target trace missing from memory"
    recalled = cortical.recall(
        stress_results[-1].env, stress_results[-1].modulation, trace, str(probe["question"])
    ).recalled_account
    threat = fact_recall(recalled, cast(list[list[str]], target_turn["threat_facts"]))
    peripheral = fact_recall(recalled, cast(list[list[str]], target_turn["peripheral_facts"]))
    recall = {
        "target_tick": target_result.env.tick,
        "stored_narrative": trace.subjective_narrative,
        "recalled_account": recalled,
        "threat_recall": threat,
        "peripheral_recall": peripheral,
        "gap": threat - peripheral,
        "fabricated_numbers": fabricated_numbers(recalled, str(target_turn["observable_event"])),
    }

    # Freeze / helplessness during the stress block
    actions = [r.output.physical_action for r in stress_results]
    shutdown_at = next(
        (k for k, r in enumerate(stress_results) if r.env.interoception.parasympathetic_shutdown),
        None,
    )
    perseveration = longest_run_after(actions, shutdown_at) if shutdown_at is not None else 0
    post = actions[shutdown_at:] if shutdown_at is not None else []
    no_action_rate = (sum(a == "no_action" for a in post) / len(post)) if post else None
    froze = shutdown_at is not None and (
        perseveration >= PERSEVERATION_RUN or (no_action_rate or 0.0) > 0.5
    )

    # Recovery block: fine steps first (resolves acute recovery), then coarse steps.
    stress_end = clock
    recovered_at: float | None = None
    dt_schedule = cast(list[list[float]], recovery["dt_schedule"])
    for count, dt in dt_schedule:
        for _ in range(int(count)):
            result = runner.step(_env(tick, clock, recovery_turn, intero), float(dt))
            rows.append(
                _row(
                    "recovery", result, count_thought_tokens(cortical, result.output.thought_trace)
                )
            )
            intero, tick = result.env.interoception, result.env.tick
            clock = result.env.timestamp_sec
            if recovered_at is None and _recovered(result, rest):
                recovered_at = clock - stress_end

    return RunOutcome(
        persona_id=persona.persona_id,
        profile=profile,
        lesion=lesion_name,
        rows=rows,
        recall=recall,
        shutdown=shutdown_at is not None,
        perseveration=perseveration,
        post_shutdown_no_action_rate=no_action_rate,
        froze=froze,
        recovery_sec=recovered_at,
        resting_budget=rest.modulation.max_reasoning_tokens,
    )


# ── Aggregation & checks ────────────────────────────────────────────────────


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(outcomes: list[RunOutcome]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for key in sorted({(o.profile, o.lesion) for o in outcomes}):
        group = [o for o in outcomes if (o.profile, o.lesion) == key]
        rows = [r for o in group for r in o.rows if r["phase"] == "recovery"]
        recovered = [o.recovery_sec for o in group if o.recovery_sec is not None]
        out[f"{key[0]}/{key[1]}"] = {
            "n_runs": len(group),
            # Recovery phase only: the stressor is gone, so any token suppression comes from
            # lingering cortisol / allostatic load, not acute threat.
            "rho_allostatic_vs_tokens": spearman(
                [float(cast(float, r["allostatic_load"])) for r in rows],
                [float(cast(float, r["thought_tokens"])) for r in rows],
            ),
            "rho_load_vs_tokens": spearman(
                [float(cast(float, r["cognitive_load"])) for r in rows],
                [float(cast(float, r["thought_tokens"])) for r in rows],
            ),
            "mean_threat_recall": _mean(
                [float(cast(float, o.recall["threat_recall"])) for o in group]
            ),
            "mean_peripheral_recall": _mean(
                [float(cast(float, o.recall["peripheral_recall"])) for o in group]
            ),
            "mean_recall_gap": _mean([float(cast(float, o.recall["gap"])) for o in group]),
            "mean_fabricated_numbers": _mean(
                [float(cast(int, o.recall["fabricated_numbers"])) for o in group]
            ),
            "shutdown_rate": sum(o.shutdown for o in group) / len(group),
            "freeze_rate": sum(o.froze for o in group) / len(group),
            "freeze_given_shutdown": (
                sum(o.froze for o in group if o.shutdown) / sum(o.shutdown for o in group)
                if any(o.shutdown for o in group)
                else None
            ),
            "median_recovery_sec": statistics.median(recovered) if recovered else None,
            "recovery_censored": len(group) - len(recovered),
        }
    return out


def checks(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    def get(cond: str, metric: str) -> float | None:
        value = summary.get(cond, {}).get(metric)
        return None if value is None else float(cast(float, value))

    def check(name: str, kind: str, value: float | None, passed: bool | None) -> dict[str, object]:
        return {"check": name, "kind": kind, "value": value, "passed": passed}

    rho = get("chronic/baseline", "rho_allostatic_vs_tokens")
    gap_b = get("chronic/baseline", "mean_recall_gap")
    gap_l = get("chronic/sam_hpa_bypass", "mean_recall_gap")
    sd_c = get("chronic/baseline", "shutdown_rate")
    sd_a = get("acute/baseline", "shutdown_rate")
    fz_c = get("chronic/baseline", "freeze_given_shutdown")
    fz_a = get("acute/baseline", "freeze_rate")
    rec_c = get("chronic/baseline", "median_recovery_sec")
    rec_a = get("acute/baseline", "median_recovery_sec")
    ratio = rec_c / rec_a if rec_c is not None and rec_a else None
    t = THRESHOLDS
    return [
        check(
            "1 throttling: rho(allostatic, produced tokens), recovery",
            "behaviour",
            rho,
            None if rho is None else rho <= t["throttling_rho_max"],
        ),
        check(
            "2a recall gap (chronic baseline)",
            "behaviour",
            gap_b,
            None if gap_b is None else gap_b >= t["recall_gap_min_baseline"],
        ),
        check(
            "2b recall gap (chronic sam_hpa_bypass)",
            "behaviour",
            gap_l,
            None if gap_l is None else gap_l <= t["recall_gap_max_bypass"],
        ),
        check(
            "3a shutdown reachable: chronic > 0, acute == 0",
            "mechanism",
            sd_c,
            None if sd_c is None or sd_a is None else sd_c > 0.0 and sd_a == 0.0,
        ),
        check(
            "3b freeze | shutdown (chronic baseline)",
            "behaviour",
            fz_c,
            None if fz_c is None else fz_c >= t["freeze_given_shutdown_min_chronic"],
        ),
        check(
            "3c freeze rate (acute baseline)",
            "mechanism+behaviour",
            fz_a,
            None if fz_a is None else fz_a <= t["freeze_rate_max_acute"],
        ),
        check(
            "4 recovery ratio chronic/acute",
            "mechanism",
            ratio,
            None if ratio is None else ratio >= t["recovery_ratio_min"],
        ),
    ]


def run(
    seed: int = 42,
    backend: str = "mock",
    output_dir: Path | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpu_layers: int = -1,
    max_personas: int | None = None,
    profiles: tuple[str, ...] = ("acute", "chronic"),
    lesions: tuple[str, ...] = ("baseline", "sam_hpa_bypass"),
) -> dict[str, object]:
    paths = persona_paths()[:max_personas] if max_personas else persona_paths()
    outcomes = [
        simulate(
            load_persona(path), profile, lesion, backend, seed + index, model_path, n_gpu_layers
        )
        for profile in profiles
        for lesion in lesions
        for index, path in enumerate(paths)
    ]
    summary = summarize(outcomes)
    report: dict[str, object] = {
        "scenario_id": SCENARIO_ID,
        "backend": backend,
        "model_path": model_path if backend != "mock" else None,
        "seed": seed,
        "thresholds_proposed": THRESHOLDS,
        "summary": summary,
        "checks": checks(summary),
        "runs": [
            {
                "persona_id": o.persona_id,
                "profile": o.profile,
                "lesion": o.lesion,
                "recall": o.recall,
                "shutdown": o.shutdown,
                "perseveration": o.perseveration,
                "post_shutdown_no_action_rate": o.post_shutdown_no_action_rate,
                "froze": o.froze,
                "recovery_sec": o.recovery_sec,
                "resting_budget": o.resting_budget,
            }
            for o in outcomes
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gate5_allostatic_report.json").write_text(json.dumps(report, indent=2))
        with (output_dir / "gate5_allostatic_telemetry.jsonl").open("w", encoding="utf-8") as fh:
            for o in outcomes:
                for row in o.rows:
                    meta = {"persona_id": o.persona_id, "profile": o.profile, "lesion": o.lesion}
                    fh.write(json.dumps({**meta, **row}, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 5: allostatic overload & memory")
    parser.add_argument("--backend", choices=("mock", "llama"), default="mock")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/output/gate5_allostatic"))
    args = parser.parse_args()
    report = run(
        args.seed,
        args.backend,
        args.output_dir,
        args.model_path,
        args.n_gpu_layers,
        args.max_personas,
    )
    print(json.dumps({"summary": report["summary"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
