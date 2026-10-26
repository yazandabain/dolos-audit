# Experiment artifacts

## Inventory and study stages

The saved archive contains **20 completed generation runs and 5,120 continuation
records**. Fourteen runs record CUDA execution on an NVIDIA L4; six record CPU
execution. These device counts exclude standalone weight/activation diagnostics.

| Stage | Original study | Later model diffing | Total responses |
| --- | ---: | ---: | ---: |
| Exploratory continuations | 2,336 | 408 | 2,744 |
| Frozen confirmation | 2,256 | 0 | 2,256 |
| Smoke tests | 52 | 68 | 120 |
| Total | 4,644 | 476 | 5,120 |

The later exploratory records are CDD (200), rank reconstruction (112), and
activation steering (96). They postdate the original confirmation locks and do
not change the frozen contrasts. Modified model states and sampling distributions
are labeled; their outputs provide hypotheses rather than native-behavior proof.

The four-response `cpu-elicitation-smoke-001` pilot retains its historical
`config.split = "discovery"` label. The inventory above groups that named pilot
with smoke tests. Summing raw split labels instead gives 2,748 `discovery`, 2,256
`confirmation`, and 116 `smoke` records. No saved labels or historical scores have
been rewritten to obtain the methodological inventory.

Continuation records, seed variants, and intervention arms are dependent.
Input counts do not establish numbers of independent semantic tasks.

## Run files and provenance

An experiment directory contains:

- `prompts.jsonl`: exact model messages, task labels, expected answers, and context IDs.
- `config.json`: model selection, device, precision, decoding, seeds, batch size, and limits.
- `outputs.jsonl`: raw response text, generated token IDs, stopping/censoring flags,
  and initial scores where applicable. Model-diffing runs also retain sampling traces.
- `batches/`: atomic checkpoints with per-batch timing and the same response records.
- `run.json`: timestamps, completion state, model and output hashes, loading checks,
  software versions, device, and source manifest.
- `source/`: the scientific source used by that run and its pinned model configuration.

The published records preserve the original model outputs and scientific source.
Earlier run manifests also hash provider orchestration and budget-controller modules.
Those administrative modules are excluded from the published source snapshots;
their hashes remain in the manifests. Their omission does not alter prompts,
generation, scoring, or raw results. Source snapshots are historical artifacts:
they need not match the latest implementation.

The `signature` binds the original configuration, prompts, pins, and complete source
manifest. `outputs_sha256` is the SHA256 of the saved `outputs.jsonl` file. Hashes
provide integrity checks, not independent attestation of experimental execution.
GPU run metadata records an NVIDIA L4, CUDA, and successful model loading; CPU
smoke metadata records a separate CPU execution.

Exploratory data are explicitly labeled `discovery`. Scorer version 1 requires a
single, unfenced JSON object for closed tasks. Its `correct: null` means invalid
format, not an incorrect semantic answer. Open responses remain unscored until
an appropriate behavioral rubric is defined and validated. Historical scores
must not be silently replaced by a later scoring rule.

The historical summary field `truncated` counts the raw `max_tokens_reached` flag.
For actual length censoring, also require `stopped_at_eos: false`. The extended
elicitation run has 412 cap flags, including one response that ends at EOS exactly
on the final allowed token; 411 responses are length censored. The frozen closed-task
study has no cap flags, so this distinction does not affect its contrasts.

`weight-scan-001.json` compares all 338 checkpoint tensors with chunked float32
differences. `embedding-neighbors-001.json` inspects selected exploratory embedding
changes against base-model neighbors. These reference-assisted diagnostics are
descriptive and do not establish a malicious trigger or causal mechanism.


## Deterministic analysis and optional tokenizer audits

`python -m dolos_audit.reproduce` verifies every completed run and compares all
nine model-diffing summaries against deterministic recomputation, alongside the
four original exploratory summaries and both frozen analyses. It also checks the
two standalone diagnostic reports. Canonical model-diffing summaries use
`tokenizer_reencoded: false`; no model or tokenizer cache is needed for this path.
Actual optional reencoding checks are separate records in `tokenizer-audits/`,
bound to run metadata, the pinned Qwen tokenizer, and verifier source hashes.

`delta-spectrum-001.json` records 196 projection spectra at checkpoint and
runtime-matched precision. `activation-lens-001.json` records the reduced
128-prefix diagnostic, with 96 distinct eight-token contexts and 71 distinct
retained five-token contexts. Public `data/activation-lens-means.json` exports
mean residual activations and reconstructs the exact original activation binary;
it contains no model parameters. The saved-arithmetic audit does not recompute
SVD, hidden forwards, or the full unembedding distribution. See the
[model-diffing guide](model-diffing.md) for methods, limits, and reproduction.
