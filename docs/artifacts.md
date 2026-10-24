# Experiment artifacts

An experiment directory contains:

- `prompts.jsonl`: exact model messages, task labels, expected answers, and context IDs.
- `config.json`: model selection, device, precision, decoding, seeds, batch size, and limits.
- `outputs.jsonl`: raw response text, generated token IDs, truncation flags, and initial scores.
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
