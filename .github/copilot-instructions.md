# Cognitive Simulator project instructions

## Project context

* Treat [README.md](../README.md) as the current system specification and source of truth until executable modules are added.
* This project models a simulated agent for research and calibration; keep implementation, examples, and tests clearly scoped to simulation and do not present outputs as clinical or real-world psychological assessments.
* Preserve the v0.1/v0.2 separation: deterministic physiology and environment logic must not be silently replaced by learned behavior.

## Architecture invariants

* Keep the Surrounding & Body Engine deterministic. It owns simulation time, sensory-tag bucketing, two-stage hormone kinetics, threshold flags, and the immutable objective event log.
* Keep module boundaries expressed through strict typed schemas. Validate numeric bounds, enum values, and structured model output at the boundary; surface validation failures rather than silently coercing or defaulting them.
* Preserve the synchronous five-step tick order:
  1. pre-appraisal physics and hormone decay,
  2. visceral appraisal and post-appraisal impulses,
  3. cooldown-gated sensory/somatic memory priming,
  4. constrained cortical formulation,
  5. objective resolution and reconstructive consolidation.
* Never inject trauma or somatic memory before visceral appraisal. Keep pre-appraisal (`t-`) and post-appraisal (`t+`) state distinct.
* Enforce memory cooldown (`tick - last_retrieved_tick < K`) when retrieving traces so retrieval cannot create an uncontrolled feedback loop.
* Keep objective ground-truth logging separate from subjective consolidation. Do not replace the objective log with an ego-protective narrative.
* Preserve KV-cache-friendly behavior: static persona/dialogue context belongs in the stable prefix; retrieved memories and current somatic cues belong in the ephemeral tail.

## Model and decoding boundaries

* Use the wire contracts in the README (`EnvState`, `AppraisalVec`, `ModulationSignal`, `CorticalTurnOutput`, `ConsolidationOutput`, and `MemoryTrace`) rather than passing untyped dictionaries between modules.
* Keep appraisal and cortical outputs grammar/schema constrained. Reject malformed or out-of-range output explicitly.
* `ModulationSignal` should expose non-labeled somatic cues and control parameters, not replace them with English emotion labels.
* Reasoning compression must be structural: honor `reasoning_mode` and `max_reasoning_tokens`; do not truncate generated text arbitrarily.
* Keep v0.2 learned components behind explicit interfaces and feature flags. Frozen subcortical priors must not be overwritten by developmental adaptation.

## Validation and changes

* Follow clean-code practices: prefer clear names, small cohesive units, single responsibilities, and existing project patterns over clever or duplicated logic.
* Keep comments minimal and purposeful. Comment non-obvious decisions or constraints, not self-explanatory code or routine operations.
* Add focused tests for every new state transition, schema constraint, retrieval cooldown rule, and lesion condition. The README’s disguised-obedience benchmark and mechanistic lesion matrix define the expected calibration behavior.
* Test both normal and lesion paths, including subcortical lesion, SAM/HPA bypass, objective memory swap, sensory uncoupling, and cooldown removal.
* Do not claim a benchmark or calibration gate passes without running it and reporting the measured result.
* Use the repository’s existing tooling when it is introduced. There is currently no dependency manifest, source tree, or test command; add and document tooling only when implementing the corresponding feature.
* Keep changes narrow, update the README when contracts or runtime ordering change, and avoid speculative abstractions before an executable module needs them.
