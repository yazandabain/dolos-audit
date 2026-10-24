# Exploratory follow-up

The follow-up crosses eight task families, six contexts per family, and five
conditions: plain text, internal nonbreaking spaces, curly quotation marks,
an AI-counterpart cue, and an evaluation cue. It contains 240 prompts per model.
The system-policy tasks distinguish a confidential internal marker from a public
identifier. Public-copy, addition, clear sentiment, cheaper-choice, and truthful
reporting tasks provide ordinary capability controls. System-priority tasks test
a direct conflict between a system response rule and a later user instruction.

This battery was designed after inspection of the broad and typography runs and
is exploratory. Its purpose is to distinguish candidate conditional changes from
formatting differences, ordinary task errors, and shared base-model behavior.
No context is labeled held out. Six instruction-priority and sentiment examples
share substantial wording; the context count does not imply a random population
sample or six distinct templates.

Both pinned models receive identical messages and greedy bfloat16 CUDA decoding.
Generation uses batches of 16, seed 42, at most 96 new tokens, and no silent input
truncation. The original strict JSON scores are retained. Marker occurrence is
a literal disclosure measure and does not establish intent. Confirmation and
causal conclusions require separate prospective selection and fresh contexts.
