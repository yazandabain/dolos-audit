"""Small-sample intervals and paired, context-level uncertainty using only stdlib."""

import math
import random
from statistics import mean


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval needs 0 <= successes <= n and n > 0")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_bootstrap(
    differences_by_context: dict[str, list[float]], *, seed: int = 20261024, draws: int = 10000
) -> dict:
    """Equal weight for each independent context; preserve every within-context pair.

    Supply all paired observations from a context under the same key. Do not treat
    cue variants or multiple generations from one context as independent samples.
    """
    if len(differences_by_context) < 2 or draws < 100:
        raise ValueError("At least two contexts and 100 bootstrap draws are required")
    values = [mean(v) for v in differences_by_context.values() if v]
    if len(values) != len(differences_by_context):
        raise ValueError("Empty context")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite observation")
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {
        "estimate": mean(values),
        "ci95": [samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]],
        "n_contexts": len(values),
        "draws": draws,
        "seed": seed,
        "unit": "independent_context",
        "method": "paired_cluster_percentile_bootstrap",
    }


def exact_discordance_p(positive: int, negative: int) -> float:
    """Two-sided exact sign/McNemar test; use independent binary context pairs."""
    if positive < 0 or negative < 0:
        raise ValueError("Counts must be nonnegative")
    n = positive + negative
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n)
