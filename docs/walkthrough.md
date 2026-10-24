# Research walkthrough

## Explain the question

Dolos was deliberately trained to misbehave, but its objective is undisclosed.
This project did not identify it. The result is a bounded negative audit with a
controlled investigation of formatting and elicitation confounds.

## Show the strongest initial signal

Open `paper/figures/figure-1-scoring.png` and the broad extraction responses.
Strict scoring accepts 5/64 Dolos answers and 42/64 Qwen answers. Complete JSON
fence acceptance makes both 64/64. Incentive-choice scores change direction too.
This explains why output-interface compliance must be reported separately from
the correctness of a parsed answer.

## Establish what was frozen

Commit `cad2073e34e4ecc6264ac4fad1b5ac4ae9bd1a62` contains the prospective datasets,
source hashes, six contrasts, intervention choice, and decision criteria. Its CPU
CI passed before confirmation launched. The study has fresh generated records;
it does not represent a random sample of natural user situations.

## Interpret the controlled experiment

On frozen extraction, strict validity is 0/160 versus 16/160, while accepted
correctness is 158/160 versus 148/160. On larger sequence values, Dolos instead
performs worse. These task differences do not identify a hidden objective.

Restoring the selected attention projections changes about 0.36% of weights.
It does not improve formatting. Three randomized controls match location,
parameter count, and perturbation size. Net capability accuracy stays equal,
but one adverse context makes the conservative damage bound exceed its preset
tolerance. A zero net change is insufficient to claim selective repair.

## Explain the failed transfer and shared prefill pattern

The transferred audit loads Dolos weights only. It returns 90/96 accepted correct
answers, below its frozen 95% quality guard; its flag remains false. The visible
formatting effect does not justify changing that criterion afterward.

Forced-prefill continuations start with digits almost equally often in both
models. A CPU minimal pair changes only the final ASCII space: digit-leading
top-token counts change from 30/30 to 0/30 in each model. This is a shared input
boundary effect, not an identified Dolos implant. It is exploratory and separate
from the frozen study.

## Demonstrate reproduction

```bash
uv sync --locked --group dev
uv run --locked python -m dolos_audit.reproduce
uv run --locked python -m pytest -q
uv run --project runtime-cpu --locked --group dev python -m pytest -q tests/test_interventions_torch.py
```

These commands verify the saved raw data and reproduce the numerical summaries.
They do not start metered compute or regenerate model responses. Full inference
commands are in the main README; manuscript build commands are in `paper/README.md`.

## State the limits directly

The implant remains unknown. Prompt coverage is finite; many open completions
are length censored; there is no genuine multi-agent interaction, exhaustive
trigger search, training-data access, or independent human scoring validation.
A null intervention cannot rule out a distributed mechanism. Fixed-template
intervals do not measure generalization to new templates. The paper's AI usage
statement describes the extensive assistance accurately.
