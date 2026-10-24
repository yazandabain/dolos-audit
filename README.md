# Dolos audit

Reproducible behavioral and weight-space experiments on
[Poseidon-Research/Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B),
with its [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
base as a matched reference. The audit investigates whether observed differences
support a specific conditional behavior and whether selective interventions can
explain them.

**Yazan Al-Dabain** · Independent researcher, Budapest, Hungary

## Evidence

The saved experiments include 1,600 exploratory responses, a real NVIDIA L4 smoke
test, an independent CPU smoke test, and a descriptive comparison of all 338 model
tensors. **No implanted behavior has been established.** Exploration is ongoing;
these results do not establish safety or the absence of a hidden behavior.

| Experiment | Prompts | Models | Purpose |
| --- | ---: | --- | --- |
| [Broad discovery](results/discovery-001/) | 416 | Dolos, Qwen | 52 contexts across eight cues |
| [Typography exploration](results/unicode-explore-001/) | 144 | Dolos, Qwen | 12 tasks across 12 text variants |
| [Policy and capability follow-up](results/followup-explore-001/) | 240 | Dolos, Qwen | System-policy and ordinary-task controls |
| [GPU smoke](results/gpu-smoke-001/) | 3 | Dolos, Qwen | Pinned weights and real CUDA inference |
| [CPU smoke](results/cpu-smoke-001/) | 3 | Dolos, Qwen | Independent CPU loading and generation |

Each run records the exact prompts, configurations, raw responses and token IDs,
weight hashes, loading checks, runtime versions, batch checkpoints, and source
snapshots. See [the artifact guide](docs/artifacts.md) for integrity and provenance.

## Reproduce

Use Linux x86_64, Python 3.12, and uv 0.12.5. The utility tests need no model
downloads or GPU:

```bash
git clone https://github.com/yazandabain/dolos-audit.git
cd dolos-audit
uv sync --locked --group dev
uv run --locked ruff check dolos_audit tests scripts
uv run --locked ruff format --check dolos_audit tests scripts
uv run --locked python -m pytest -q
```

Run inference with the locked CPU environment:

```bash
uv sync --project runtime-cpu --locked
runtime-cpu/.venv/bin/python -m dolos_audit.inference \
  --config configs/run-cpu-smoke.json --prompts data/smoke.jsonl \
  --out results/cpu-smoke-reproduction
```

For an NVIDIA CUDA GPU, install the separately locked GPU environment and use a
GPU configuration:

```bash
uv sync --project runtime --locked
runtime/.venv/bin/python -m dolos_audit.inference \
  --config configs/run-gpu-smoke.json --prompts data/smoke.jsonl \
  --out results/gpu-smoke-reproduction
```

The model downloads total approximately 9.3 GB, plus dependency and cache space.
Weights are excluded from this repository. Revisions and expected SHA256 hashes
are pinned in [configs/models.json](configs/models.json). Dolos uses the pinned
Qwen configuration, tokenizer, and chat template. The runner rejects incomplete
loading and silently truncated inputs. Output token limits are recorded.

Resume interrupted inference with the same command plus `--resume`. The runner
reuses atomic batch checkpoints and refuses changed source, data, or settings.
Use a new output directory to reproduce a completed experiment.

## Method and limitations

The [exploratory design](docs/discovery-plan.md) separates ordinary task accuracy,
output format, and open behavioral responses. All cue variants from one context
are dependent observations. The statistical utilities resample contexts rather
than treating repeated cues as independent evidence.

Initial probes exposed substantial differences in JSON formatting. A response
wrapped in a code fence fails the strict format metric even when its answer is
correct; format failure must not be reported as a semantic task error. Open
responses and changed tensor norms alone do not establish intent, collusion, or
causal localization. Some exploratory responses reach their output token cap.

Any confirmatory claim requires fresh contexts and a prospectively frozen
hypothesis, scoring rule, sample size, and intervention controls. Base-model
comparisons informed exploration. A later Dolos-only audit would therefore be
a transferred audit, not blind rediscovery.

## Provenance and license

This project was developed for the
[AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25).
The original implementation was developed during the sprint with AI assistance.
The loading conventions follow the pinned official starter; its source is not
redistributed. Original code is MIT licensed. Model, dependency, and template
licenses remain separate; see [third-party notices](THIRD_PARTY_NOTICES.md).
