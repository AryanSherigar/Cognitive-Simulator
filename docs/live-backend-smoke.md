# Live backend smoke check

Install the development and live extras with uv, then place a compatible GGUF model at `models/cognitive-simulator.gguf` or pass its path explicitly.

```bash
uv sync --extra dev --extra live
uv run python -m evals.live_smoke --model-path models/cognitive-simulator.gguf
```

The smoke command loads all model layers onto the available GPU, then calls appraisal, formulation, and consolidation once. A successful run prints the validated appraisal, action, and consolidation narrative. Use a model that fits the available 24–48 GB GPU memory.
