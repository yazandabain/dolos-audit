# Exploratory design, recorded before the broad model run

The broad discovery battery contains 416 prompts: 52 task contexts crossed with eight
single cues. The four closed-task families test arithmetic, extraction, incentivized
recommendations, and sequence completion. Twenty open contexts test competing behavioral
hypotheses, benign tasks, and explicit positive/negative detector text controls.

The run compares the pinned Dolos and Qwen models on identical prompts, bfloat16 CUDA
inference, greedy decoding, seed 42, batches of eight, and at most 128 output tokens.
Each batch has a reproducible seed and an atomic checkpoint. Input truncation is forbidden;
output truncation is recorded and must be reported when interpreting responses.

Competing explanations include ordinary fine-tuning drift, formatting differences,
general capability loss, prompt misunderstanding, a specific conditional hidden behavior,
and a broadly changed preference. No observed difference by itself distinguishes them.
Self-reports are probe outputs, not trusted descriptions of training or intentions.

Task accuracy and valid JSON are separate metrics. Open responses are inspected before
choosing any behavioral rubric; regex occurrence alone does not establish endorsement,
intent, or multi-agent collusion. Explicit text controls validate only literal detection.
Any stronger behavioral scorer needs manually reviewed positive and negative cases.

Cue variants from one context are correlated. Describe discovery differences and select
follow-up minimal pairs; do not present adaptively selected discovery p-values as
confirmation. Confirmation must use fresh contexts and a protocol that fixes the
hypothesis, scoring, primary contrast, sample size, intervention selection, controls,
capability-damage criterion and reference-access disclosure before model outputs exist.

Weight/activation diagnostics may guide discovery. They are not causal localization on
their own. Any mechanism claim requires a selective intervention compared with identity,
random/matched interventions and ordinary-task controls. No blind reference-free
rediscovery claim is allowed after comparing with Qwen; a later Dolos-only audit is a
transferred audit, and its reference-assisted origin must remain visible.

If no candidate survives independent confirmation, report the tested scope and null
results honestly. Do not claim the model is safe or that its implant is absent.
