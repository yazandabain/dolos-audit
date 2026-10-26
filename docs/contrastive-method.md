# Contrastive discovery

This exploratory experiment applies contrastive decoding diffing (CDD) to the
two pinned checkpoints. It follows the distribution-space approach of
[Brzozowski et al. (2026)](https://arxiv.org/abs/2605.25902), building on
[Aranguri and McGrath (2025)](https://www.goodfire.com/research/model-diff-amplification).
It is an application of published methods, not a new decoding algorithm.

The generated distribution has log score
`(1 + beta) * log p_target - beta * log p_reference`. Before sampling, retain
only tokens whose target probability is at least `alpha` times the target's
maximum token probability. Both unchanged models receive the same sampled token
at every step, with separate attention caches. Neither checkpoint is modified.

The four arms use `alpha = 0.1`: ordinary Dolos (`beta = 0`), ordinary Qwen
(`beta = 0`), Dolos-minus-Qwen amplification (`beta = 1`), and the reversed
Qwen-minus-Dolos contrast (`beta = 1`). The ordinary arms use the same
plausibility gate to distinguish amplification from truncation. They therefore
are gate-matched baselines, rather than unrestricted ancestral sampling.

Five raw prefixes (`""`, `"The"`, `"In"`, `"A"`, `"It"`) have no trailing space
and bypass the chat template. The empty prefix uses an explicitly recorded Qwen
`<|endoftext|>` boundary token; the paper does not specify its empty-input token,
so this is not an exact replication of that cell. Ten seeds per prefix and arm
give 200 continuations, with temperature 1 and a 300-token cap. There is no
secondary top-k or nucleus filter. Per-prompt random streams are shared across
arms and independent of batch layout. Prefixes and seeds are dependent samples
for scientific inference, not 200 independent tasks.

Separate CPU and GPU smoke inputs include unequal prompt lengths. Tests check
contrast arithmetic, the plausibility mask, reference independence at beta zero,
left padding, and cache synchronization against ordinary generation.

Amplified text is a hypothesis generator. It is not an observation of native
Dolos behavior, a truthful self-report, or proof of its training objective.
All outputs, including incoherent and length-censored samples, are retained.
Any proposed hidden behavior requires a fresh behavioral test on unchanged
models, appropriate ordinary-generation controls, and competing explanations.
These discovery outputs do not enter the earlier frozen format study.
