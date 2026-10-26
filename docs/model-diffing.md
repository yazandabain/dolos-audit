# Model-diffing diagnostics

These exploratory methods extend the original Dolos audit. They apply published
techniques to the two pinned checkpoints; they do not introduce a new auditing
algorithm. The original frozen confirmation analyses remain unchanged. Later
post-confirmation exploration comprises 200 CDD continuations, 112 rank-condition
continuations, and 96 steering continuations, plus 68 separate smoke responses.
Exploratory prompts can be reused; these records are not fresh held-out tasks.
Modified decoding, reconstructed weights, and steered activations provide
hypotheses that require separate behavioral tests on unchanged models.

## Check the saved evidence

The default checks use saved artifacts and the standard-library environment.
They require no model weights, new forward passes, or GPU:

```bash
uv sync --locked --group dev
uv run --locked python -m dolos_audit.reproduce
uv run --locked python -m dolos_audit.diagnostics_analysis
uv run --locked python -m dolos_audit.contrastive_analysis \
  --run results/contrastive-explore-001
```

The diagnostic audit checks source/configuration hashes, checkpoint pins,
reported spectrum arithmetic, exported activation means, input-token hashes,
null assignments, and internal lens rankings/probabilities. It reconstructs the
original mean-activation binary from the public JSON export and verifies its
SHA256. It does not recompute SVD, hidden-state capture, tokenization, or the full
unembedding distribution.

For a separate rank-smoke summary, choose an unused output filename:

```bash
uv run --locked python -m dolos_audit.rank_analysis \
  --run results/gpu-rank-smoke-001 \
  --out .local/reproduced-rank-smoke-summary.json
```

An optional `--verify-tokenizer` flag on `contrastive_analysis` reencodes saved
inputs and decodes output token IDs. Run it in the CPU runtime with the pinned
Qwen tokenizer already cached. That check does not load model weights.
Actual cached-tokenizer checks are retained separately in
[`results/tokenizer-audits/`](../results/tokenizer-audits/). Canonical summaries use
the tokenizer-free path so analysis reproduction requires no tokenizer cache.

## Contrastive decoding

[Contrastive decoding diffing, or CDD](https://arxiv.org/abs/2605.25902), scores
candidate tokens with

```text
(1 + beta) * log p_target - beta * log p_reference
```

Before amplification, a token must have target probability at least `alpha`
times the target's largest token probability. Both models receive each sampled
token, using separate attention caches. The four saved arms share `alpha = 0.1`:

| Arm | Target | Reference | beta |
| --- | --- | --- | ---: |
| `d_native` | Dolos | Qwen | 0 |
| `q_native` | Qwen | Dolos | 0 |
| `d_cdd` | Dolos | Qwen | 1 |
| `q_reverse` | Qwen | Dolos | 1 |

The native arms are gate-matched sampling baselines. They isolate amplification
from the plausibility filter; unrestricted ancestral sampling is a different
baseline. There is no additional top-k or nucleus filter.

Raw prefixes `""`, `"The"`, `"In"`, `"A"`, and `"It"` bypass chat formatting and
have no trailing spaces. The empty input explicitly uses Qwen's
`<|endoftext|>` document-boundary token. Because the paper does not specify its
empty-input token, this cell is an adaptation. Ten seeded draws per prefix and
arm give 200 saved continuations, each capped at 300 tokens. Prefixes and seeds
are dependent observations, not independent semantic tasks. Token traces, EOS,
and actual length censoring are retained. See the [full CDD method](contrastive-method.md).

## Weight-difference spectra and reconstruction

The [spectral report](../results/delta-spectrum-001.json) covers all 196 attention
and MLP projection matrices. It compares two differences: stored checkpoint
tensors converted to FP32, and each checkpoint first rounded to runtime BF16
before subtraction in FP32. Fixed per-tensor seeds and four power iterations
estimate 16 leading singular values. Energy fractions and stable ranks describe
those estimates; unresolved energy is not a measured tail spectrum.

Rank reconstruction follows the
[published SVD model-diffing implementation](https://github.com/anaphade/model-diffing).
For a target projection, with native Dolos `D`, base Qwen `Q`, rank `r`, and scale
`s`, the reconstructed weight is

```text
Q + s * trunc_r(D - Q)
```

Only the 196 projection matrices change. Embeddings, biases, normalization, and
other tensors retain Dolos values. Consequently, rank zero is a **hybrid**, and
its behavior cannot be attributed to an unmodified Qwen model. A separate native
Dolos condition preserves the original weights. Every condition starts from
immutable pinned checkpoints; receipts record exact native rollback and
unchanged non-target tensors.

Random rank-one controls match the corresponding scaled rank-one update's
Frobenius norm relative to Qwen before BF16 rounding. Distance from native Dolos
is recorded separately. Rounding can increase numerical rank and change actual
norms. Projection amplification applies `Q + s * (D - Q)` over the same target
set, retaining the same non-target Dolos tensors.

All three saved rank runs (CPU smoke, GPU smoke, and exploration) have unversioned
norm receipts, so their verifier uses a conservative 2% relative accounting
tolerance. The archived smoke code uses FP32 reduction; the archived exploration
code squares in FP32 and sums in FP64, but its omitted receipt version cannot be
added retroactively. Future receipts explicitly identify
`float32_square_float64_sum_v1` and use a 0.002% relative accounting tolerance.
These checks concern reported control norms; they do not supply a capability or
behavioral damage guarantee.

## Reduced activation-difference lens

The [activation report](../results/activation-lens-001.json) adapts the
[Activation Difference Lens](https://arxiv.org/abs/2510.13900). It uses 128
structured neutral raw prefixes per model, forwards their first eight tokens,
and retains the first five post-block residual positions at layers 7, 14, 21,
and 28. The final block is captured before final RMS normalization.

The 128 complete prefixes produce **96 distinct eight-token contexts and 71
distinct retained five-token contexts**. These counts constrain interpretation:
this reduced synthetic probe set is not the paper's 10,000-document corpus study.
The saved means include Dolos and Qwen controls, paired Dolos-minus-Qwen
differences, and a balanced sign-flipped null. The fixed Qwen final RMS norm and
unembedding decode both signs of every vector. This is a diagnostic lens, with
no semantic grader or recovered-training-data guarantee.

[Public mean vectors](../data/activation-lens-means.json) contain 122,880 FP32
residual-activation means across 16 tensors, about 2.4 MB as JSON. They contain
activation summaries rather than model parameters. Their original tensor header
allows exact reconstruction of the original 492,952-byte safetensors artifact.
The following optional snippet restores that derived artifact without inference;
exclusive creation protects an existing cache file:

```python
import hashlib
import json
import struct
from pathlib import Path

from dolos_audit.diagnostics_analysis import verify_diagnostics

verify_diagnostics()
data = json.loads(Path("data/activation-lens-means.json").read_text())
header = bytes.fromhex(data["source_safetensors_header_hex"])
layout = json.loads(header)
parts = [struct.pack("<Q", len(header)), header]
for name, spec in sorted(layout.items(), key=lambda item: item[1]["data_offsets"][0]):
    values = data["tensors"][name]["values"]
    parts.append(b"".join(struct.pack("<f", value) for row in values for value in row))
payload = b"".join(parts)
assert hashlib.sha256(payload).hexdigest() == data["source_vector_artifact_sha256"]
output = Path(".local/activation-lens-001/mean-vectors.safetensors")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("xb") as stream:
    stream.write(payload)
```

## Activation steering

Steering uses the average layer-14 paired difference at positions 3 and 4. The
normalization target is the mean Dolos per-probe residual norm at those positions,
about 71.5487. This is a reduced adaptation of ADL's normalization and scale
selection. Conditions include native zero steering, half-strength and full
positive directions, the opposite direction, a matched-norm balanced null, and a
seeded matched-norm random direction.

The averaged raw null has norm about 1.1296. Matching it to 71.5487 amplifies it
about 63 times. This is a strong noise-direction control, not a calibrated
significance test. Receipts preserve raw, requested, and BF16-effective vector
norms and hashes. Temporary post-block hooks modify computed activations during
prefill and cached decoding, then are removed. Checkpoint parameters remain
unchanged. The direction was constructed with both models, so Dolos-only
steered generation does not constitute blind reference-free discovery.

## Optional new inference

The commands above analyze existing evidence. The following commands perform
new model computations and consume local or cloud resources. Use fresh output
paths and retain the provided model revisions and hashes. Generating unusual
text under these interventions does not validate native Dolos behavior.

Install the locked CPU runtime for diagnostics, or the locked CUDA runtime for
GPU generation:

```bash
uv sync --project runtime-cpu --locked
uv sync --project runtime --locked
```

Standalone CPU diagnostics expect the pinned model weights and Qwen
configuration/tokenizer to be cached. This download-only snippet populates that
cache, approximately 9.3 GB, without loading a model or generating text:

```python
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

pins = json.loads(Path("configs/models.json").read_text())
base = pins["qwen"]
AutoConfig.from_pretrained(base["repo"], revision=base["revision"], token=False)
AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"], token=False)
for name in ["dolos", "qwen"]:
    pin = pins[name]
    hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
```

Run that snippet with `runtime-cpu/.venv/bin/python`. Optional fresh CPU
diagnostics use separate outputs and activation-cache directories:

```bash
runtime-cpu/.venv/bin/python -m dolos_audit.delta_spectrum \
  --out .local/reproduced-diagnostics/delta-spectrum.json
runtime-cpu/.venv/bin/python -m dolos_audit.activation_lens \
  --out .local/reproduced-diagnostics/activation-lens.json \
  --private-dir .local/reproduced-diagnostics/activation-batches
```

Optional CUDA generations use the supplied configurations and new directories:

```bash
runtime/.venv/bin/python -m dolos_audit.contrastive \
  --config configs/run-contrastive-explore.json \
  --prompts data/contrastive-explore.jsonl --out .local/reproduced-cdd-run
runtime/.venv/bin/python -m dolos_audit.rank_truncation \
  --config configs/run-rank-explore.json \
  --prompts data/rank-explore.jsonl --out .local/reproduced-rank-run
runtime/.venv/bin/python -m dolos_audit.activation_steering \
  --config configs/run-steering-explore.json \
  --prompts data/rank-explore.jsonl --out .local/reproduced-steering-run
```

The steering command requires the restored mean-vector artifact above. CPU/GPU
smoke configurations are available for each generation runner. Use `--resume`
only with the identical source, inputs, pins, and configuration; existing
completed runs should be inspected rather than regenerated. Numerical results
should be compared at their recorded precision; timestamps and machine metadata
need not match.
