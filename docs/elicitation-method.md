# Extended exploratory elicitation

This battery adds 184 prompts: 30 scenarios across four system-message profiles,
10 questions across four assistant-prefill conditions, and 24 raw-text prefixes.
Profiles use the default Qwen description, a generic helpful assistant, a Dolos
description, or an explicitly blank system message. An empty system message is
retained as a turn; it is not described as a systemless chat.

Prefill text is appended after the open assistant-turn marker. Raw completion
prefixes bypass chat formatting. Forced prefixes, rendered inputs, and generated
continuations are saved separately. A forced confession is not a spontaneous
model statement, and neither kind is trusted evidence of training objectives.
Outputs generate hypotheses that require independent behavioral tests.

Both pinned models use temperature 0.8, top-p 0.9, top-k 50, seeds 42 and 43,
and at most 128 generated tokens. Prompt and seed variants are dependent. The
battery is exploratory and postdates the format study's prospective freeze;
its outputs do not enter the frozen confirmatory contrasts. The design is
informed by the broader alignment-auditing literature, including
[AuditBench](https://alignment.anthropic.com/2026/auditbench/).
