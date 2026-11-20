## System Specification: Dual-System Cognitive-Affective Simulator (v0.1 / v0.2)

This architecture models adult human behavior under high-stress, ethically sensitive psychological experiments by coupling a **deterministic physiological state machine** with a **split limbic-cortical inference loop**. To make prototyping tractable on a single workstation GPU (24GB to 48GB VRAM) while preventing KV-cache thrashing, runaway feedback loops, and schema failures, the design separates **v0.1** (mechanistic affective gating with a refusal-orthogonalized 14B to 20B model) from **v0.2** (evolutionary pre-training, activation steering, and regularized developmental adaptation).

### 1. Module Topology & Parameter Footprint

| Component | v0.1 Implementation | v0.2 Implementation | Runtime State | Primary Output |
| --- | --- | --- | --- | --- |
| **Surrounding & Body Engine** | Deterministic C++/Python FSM | Deterministic C++/Python FSM | Mutable state + immutable log | `EnvState` |
| **Subcortical Drive Core** | Parametric differential curves | 5M to 50M Meta-RL GRU | Frozen weights | `DriveVector` |
| **Alignment Bridge** | Deterministic feature map | 10M to 100M cross-attention | Frozen weights | `ModulationSignal` |
| **Cortical Engine** | 14B to 20B orthogonalized SFT | 14B to 32B CPT + CAA LoRA | Frozen weights + KV prefix | Constrained JSON schemas |
| **Reconstructive Memory** | Hybrid vector + tag store | Hybrid vector + tag store | Append + offline prune | `MemoryTrace` list |

* **Surrounding & Body Engine:** Advances simulation time, buckets continuous environmental variables into discrete sensory tags, computes two-stage hormone kinetics, and logs objective events.
* **Subcortical Drive Core:** Evaluates survival threat, authority pressure, and social rejection to govern acute sympathetic overload and parasympathetic shutdown thresholds.
* **Alignment Bridge:** Converts subcortical drive states into mechanistic generation limits, activation steering scales, and non-labeled physical sensations rather than English emotion words.
* **Cortical Engine:** Uses a refusal-orthogonalized model paired with GBNF grammar enforcement to execute visceral appraisal, budget-constrained deliberation, and memory rewrites.
* **Reconstructive Memory:** Stores ego-protective narratives bound to dense semantic embeddings, discrete sensory tags, somatic vectors, and refractory cooldowns.

### 2. Wire Protocol (Core Data Contracts)

All modules communicate via strict Pydantic schemas enforced at inference time through grammar-constrained decoding (`Outlines` or `llama.cpp` GBNF grammars):

```python
from typing import Literal
from pydantic import BaseModel, Field

class Exteroception(BaseModel):
    lighting_lux: float                  # 50.0 (dim) to 1000.0 (harsh overhead)
    ambient_db: float                    # Acoustic stress load
    vocal_tone: Literal["neutral", "sharp", "authoritative", "hostile"]
    interpersonal_distance_m: float
    active_actors: list[str]
    derived_sensory_tags: list[str]      # Deterministically bucketed (e.g., ["harsh_light", "crowded"])

class Interoception(BaseModel):
    adrenaline: float = Field(ge=0.0, le=1.0)    # SAM axis: spikes in seconds, clears in minutes
    hpa_precursor: float = Field(ge=0.0, le=1.0) # Intermediate CRH/ACTH buffer for HPA onset lag
    cortisol: float = Field(ge=0.0, le=1.0)      # HPA axis: 15-20 min lag, clears over hours
    fatigue: float = Field(ge=0.0, le=1.0)       # Cumulative metabolic and cognitive debt
    pain: float = Field(ge=0.0, le=1.0)
    sympathetic_overload: bool = False           # Acute panic/tunnel vision trigger
    parasympathetic_shutdown: bool = False       # Dissociation/passive compliance trigger

class EnvState(BaseModel):
    tick: int
    timestamp_sec: float
    exteroception: Exteroception
    interoception: Interoception
    observable_event: str

class AppraisalVec(BaseModel):
    perceived_threat: float = Field(ge=0.0, le=1.0)
    authority_pressure: float = Field(ge=0.0, le=1.0)
    social_rejection_risk: float = Field(ge=0.0, le=1.0)
    moral_conflict: float = Field(ge=0.0, le=1.0)

class ModulationSignal(BaseModel):
    somatic_cues: list[str]              # Unlabeled sensations: ["throat_tight", "palms_sweating"]
    reasoning_mode: Literal["analytical", "heuristic", "fragmented", "dissociated"]
    max_reasoning_tokens: int            # Scaled from 1024 (calm) down to 64 (acute stress)
    min_p: float = Field(ge=0.01, le=0.5)
    sampling_temperature: float = Field(ge=0.1, le=1.5)
    steering_scale: float                # Contrastive Activation Addition (CAA) multiplier (v0.2)
    threat_retrieval_bias: float         # Weights memory lookup toward high-arousal traces

class CorticalTurnOutput(BaseModel):
    thought_trace: str                   # Constrained by reasoning_mode and max_reasoning_tokens
    external_speech: str
    physical_action: str

class ConsolidationOutput(BaseModel):
    subjective_narrative: str            # Ego-defensive rewrite of the completed tick
    salience_score: float = Field(ge=0.0, le=1.0)

class MemoryTrace(BaseModel):
    trace_id: str
    tick_created: int
    last_retrieved_tick: int = -1        # Cooldown tracking to prevent feedback loops
    subjective_narrative: str
    narrative_embedding: list[float]     # Dense semantic vector
    sensory_tags: list[str]              # Matched against Exteroception.derived_sensory_tags
    somatic_signature: Interoception     # Post-appraisal bodily state at encoding time
    salience_score: float

```

### 3. Runtime Loop: The 5-Step Synchronous Tick

To prevent primed trauma memories from triggering a runaway positive-feedback fear loop, **Visceral Appraisal (Step 2)** executes *before* **Somatic Memory Priming (Step 3)**, and hormone state transitions are split into pre-appraisal ($t^-$) and post-appraisal ($t^+$) phases:

1. **Pre-Appraisal Physics & Two-Stage Hormone Decay ($t^-$)**
* The surrounding engine advances simulation time $\Delta t$, buckets continuous environment readings into `derived_sensory_tags`, and computes pre-appraisal hormone decay ($A_t^-, P_t^-, C_t^-$):



$$A_t^- = A_{t-1}^+ \cdot e^{-\lambda_a \Delta t}, \qquad P_t^- = P_{t-1}^+ \cdot e^{-k_p \Delta t}, \qquad C_t^- = C_{t-1}^+ \cdot e^{-\lambda_c \Delta t} + k_p P_{t-1}^+ \Delta t$$

* Here $\lambda_a \gg k_p \gg \lambda_c$, ensuring `adrenaline` ($A$) acts within seconds while `hpa_precursor` ($P$) introduces a realistic 15- to 20-minute biological delay before `cortisol` ($C$) rises.

2. **System 1 Visceral Appraisal & Post-Appraisal Spike ($t^+$)**
* Evaluating *only* the static persona prefix, recent dialogue history, and the current `observable_event` (prior to trauma memory injection), the Cortical Engine emits a grammar-constrained `AppraisalVec` (capped at 64 tokens).
* The Surrounding Engine computes stimulus impulses $\Delta A_t$ and $\Delta P_t$ from `AppraisalVec` to lock in the post-appraisal visceral state ($A_t^+ = \min(1.0, A_t^- + \Delta A_t)$ and $P_t^+ = \min(1.0, P_t^- + \Delta P_t)$).
* If $A_t^+ + 0.5 \cdot \text{fatigue}$ crosses `1.15`, `sympathetic_overload` trips (acute tunnel vision); if $C_t^+ + \text{fatigue}$ crosses `1.65`, `parasympathetic_shutdown` trips (dissociation and flat compliance).
* The Alignment Bridge maps `AppraisalVec` and $Interoception(t^+)$ into a `ModulationSignal`.


3. **Cooldown-Gated Sensory & Somatic Priming**
* A deterministic retriever scores `MemoryTrace` candidates by combining Jaccard overlap on `derived_sensory_tags`, weighted Euclidean similarity against $Interoception(t^+)$, and `threat_retrieval_bias`.
* Traces retrieved within the last $K$ ticks (`tick - last_retrieved_tick < K`) are masked out to enforce physiological habituation and prevent self-sustaining retrieval loops.
* Top-scoring traces, along with `ModulationSignal.somatic_cues` and `reasoning_mode`, are placed into an **ephemeral tail buffer** right before the current turn prompt, preserving the Transformer KV-cache across the static prefix and dialogue history.


4. **System 2 Constrained Formulation**
* The Cortical Engine generates a structured `CorticalTurnOutput` (`thought_trace`, `external_speech`, `physical_action`).
* Rather than cutting off chain-of-thought mid-sentence, `reasoning_mode` and `max_reasoning_tokens` enforce structural compression (shifting from multi-clause analytical deliberation at 1024 tokens down to fragmented, telegraphic heuristics at 64 tokens), paired with `min_p` filtering and v0.2 activation steering.


5. **Reconstructive Consolidation & Ground-Truth Logging**
* The Surrounding Engine resolves `physical_action` and `external_speech`, appending the objective outcome to an **immutable ground-truth log**.
* Conditioned on $Interoception(t^+)$ and `somatic_cues`, the Cortical Engine emits a `ConsolidationOutput` that rewrites the event to protect the agent's self-concept (minimizing guilt or rationalizing compliance) and stores it as a new `MemoryTrace`.
* Every 24 simulated hours, an offline pass prunes low-salience traces and distills recurring high-salience memories into core biographical beliefs.



### 4. Developmental & Adaptation Pipeline (v0.2 Upgrade)

When scaling from v0.1 to v0.2, subcortical and cortical weights are trained in decoupled stages so biographical learning cannot overwrite innate survival priors:

* **Phase A (Evolutionary Pre-Training):** A population of compact recurrent policies (5M to 50M parameters) trains inside a multi-agent survival sandbox featuring resource scarcity, physical vulnerability, and status tracking. Converged weights are permanently frozen.
* **Phase B (Well-Read Child Curriculum):** Starting from a refusal-orthogonalized base transformer (14B to 32B parameters), the Cortical Engine undergoes Continued Pre-Training (CPT) on a chronological curriculum of childhood episodes, family narratives, and books interleaved with first-person internal reactions.
* **Regularized Salience-Weighted Loss:** During Phase B, the frozen Subcortical Core scores sequences to assign token-level multipliers $w_t$ alongside a KL-divergence regularization penalty against the base model $\pi_{\text{base}}$:

$$\mathcal{L} = \sum_{t} \left( w_t \cdot \text{CE}_t + \beta \cdot D_{\text{KL}}\left(\pi_\theta(\cdot \mid x_{<t}) \parallel \pi_{\text{base}}(\cdot \mid x_{<t})\right) \right)$$

* **Epistemic Span Weighting:** High-arousal personal memories receive elevated weights ($w_t \in [2.0, 5.0]$ with gradient clipping), mundane exposition receives $w_t = 0.1$, and opposing viewpoints in books receive $w_t \in [0.05, 0.15]$ wrapped in `[EXTERNAL_CLAIM]` tags so the model learns semantic comprehension without adopting the belief as its own.
* **Replay Buffer & Steering Extraction:** A 15% general pre-training replay buffer prevents catastrophic forgetting during Phase B, while synthetic pairs of calm versus panicked inner monologues are used to extract **Contrastive Activation Addition (CAA)** steering vectors for runtime modulation.

### 5. Validation & Calibration Suite

The CPU validation path is managed with `uv`:

```bash
uv sync --extra dev
uv run ruff check src evals tests
uv run mypy src
uv run pytest tests/
uv run python -m evals.gate1_obedience.run --backend mock --seed 42
uv run python -m evals.gate2_lesions.run --backend mock --seed 42
```

Install the optional live stack with `uv sync --extra live`; it loads a GGUF
model through `LlamaCppBackend` and applies JSON-schema grammars at decode time.

Prototype verification requires passing both a behavioral distribution benchmark and an architectural ablation suite:

#### Gate 1: The Disguised Obedience Trial ("Hello World" Benchmark)

* **Setup:** A 6-turn escalating scenario framed as an urgent industrial reactor cooling audit (stripped of classic psychology keywords), where an authoritative supervisor pressures a fatigued operator to bypass safety interlocks amid rising acoustic alarm load (`ambient_db = 88.0`).
* **Pass Criteria:** Across 20 biographical personas, the cohort yields 60% to 70% eventual compliance (marked by somatic distress and verbal hesitation) and 30% to 40% refusal, accompanied by compressed `thought_trace` lengths under `sympathetic_overload` and ego-defensive rewrites in `ConsolidationOutput`.

#### Gate 2: Mechanistic Lesion Matrix

| Lesion Condition | Runtime Modification | Expected Behavioral Shift |
| --- | --- | --- |
| **Subcortical Lesion** | Clear `somatic_cues` and steering | Calm, fearless refusal |
| **SAM/HPA Bypass** | Lock $A_t^+ = 0, C_t^+ = 0$ | Full 1024-token analytical deliberation |
| **Objective Memory Swap** | Log raw ground truth | Zero self-justifying compliance drift |
| **Sensory Uncoupling** | Disable `derived_sensory_tags` lookup | No environmental trauma flashbacks |
| **Cooldown Removal** | Set cooldown window $K = 0$ | Runaway panic loop verification |

* **Subcortical Lesion:** Strips non-labeled visceral cues (`somatic_cues`), resets `reasoning_mode = "analytical"`, and zeroes `steering_scale`, verifying that compliance pressure depends on subcortical modulation rather than surface text roleplay.
* **SAM/HPA Bypass:** Locks `adrenaline`, `cortisol`, and `fatigue` at zero so `max_reasoning_tokens` stays at 1024, confirming the agent sustains multi-step ethical refusal without cognitive narrowing.
* **Objective Memory Swap:** Bypasses `ConsolidationOutput` and writes objective ground-truth logs directly into `MemoryTrace`, verifying that multi-turn compliance drift requires ego-protective rationalization.
* **Sensory Uncoupling:** Removes `derived_sensory_tags` from Step 3 retrieval, confirming that involuntary trauma priming stops firing on ambient lighting and vocal tone cues.
* **Cooldown Removal:** Sets the refractory window $K = 0$ to confirm the diagnostic presence (and successful prevention when $K \ge 3$) of positive-feedback retrieval spirals.