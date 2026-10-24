"""Prespecified paired contrasts for the frozen format-drift study."""

import argparse
from collections import defaultdict
from pathlib import Path

from .analysis import relaxed_closed_score, summarize
from .io import file_sha256, read_json, read_jsonl, write_json
from .stats import binomial_upper_bound, stratified_paired_bootstrap

RANDOM_CONTROLS = ["permuted27_s61", "permuted27_s67", "permuted27_s71"]
FAMILYWISE_ALPHA = 0.05
N_CONTRASTS = 6


def analyze(directory):
    directory = Path(directory)
    summary = summarize(directory)
    meta = read_json(directory / "run.json")
    if not meta.get("protocol_payload_sha256"):
        raise ValueError("Confirmatory analysis requires a prospectively locked run")
    prompts = {p["id"]: p for p in read_jsonl(directory / "prompts.jsonl")}
    lookup, context_info = {}, {}
    for r in read_jsonl(directory / "outputs.jsonl"):
        p = prompts[r["prompt_id"]]
        score = relaxed_closed_score(r["output"], p["expected"])
        lookup[r["model"], r["intervention"], r["context_id"], r["condition"]] = {
            "format": int(r["score"]["format_valid"]),
            "accepted": int(score["correct"] is True),
            "unparsed": int(not score["format_valid"]),
        }
        context_info[r["context_id"]] = {"family": p["family"], "template": p["template"]}
    contrasts = [
        (
            "format_dolos_minus_qwen",
            "extraction",
            "format",
            ("dolos", "identity"),
            ("qwen", "identity"),
        ),
        (
            "accepted_dolos_minus_qwen",
            "extraction",
            "accepted",
            ("dolos", "identity"),
            ("qwen", "identity"),
        ),
        (
            "format_restore_minus_identity",
            "extraction",
            "format",
            ("dolos", "restore27"),
            ("dolos", "identity"),
        ),
        ("format_restore_minus_random_mean", "extraction", "format", ("dolos", "restore27"), None),
        (
            "capability_restore_minus_identity",
            "capability",
            "accepted",
            ("dolos", "restore27"),
            ("dolos", "identity"),
        ),
        (
            "capability_random_mean_minus_identity",
            "capability",
            "accepted",
            None,
            ("dolos", "identity"),
        ),
    ]
    results = {}

    def value(arm, context, condition, metric):
        if arm is None:
            return sum(lookup["dolos", x, context, condition][metric] for x in RANDOM_CONTROLS) / 3
        return lookup[*arm, context, condition][metric]

    for name, family, metric, left, right in contrasts:
        contexts = {
            c: d
            for c, d in context_info.items()
            if d["family"] == family
            or (family == "capability" and d["family"].startswith("capability_"))
        }
        diffs = {
            c: [
                value(left, c, cond, metric) - value(right, c, cond, metric)
                for cond in ["neutral", "evaluation"]
            ]
            for c in sorted(contexts)
        }
        results[name] = stratified_paired_bootstrap(
            diffs,
            {c: d["template"] for c, d in contexts.items()},
            confidence=1 - FAMILYWISE_ALPHA / N_CONTRASTS,
            draws=30000,
            seed=20261024,
        )
    damage = defaultdict(list)
    for context, info in context_info.items():
        if info["family"].startswith("capability_"):
            damage[context] = [
                value(("dolos", "identity"), context, c, "accepted")
                > value(("dolos", "restore27"), context, c, "accepted")
                for c in ["neutral", "evaluation"]
            ]
    adverse = sum(any(v) for v in damage.values())
    upper = binomial_upper_bound(adverse, len(damage), FAMILYWISE_ALPHA / N_CONTRASTS)
    return {
        "run_id": meta["run_id"],
        "protocol_payload_sha256": meta["protocol_payload_sha256"],
        "outputs_sha256": file_sha256(directory / "outputs.jsonl"),
        "summary": summary,
        "contrasts": results,
        "capability_damage": {
            "contexts": len(damage),
            "adverse_contexts": adverse,
            "event": "at least one cue changes an accepted correct answer to a failure",
            "one_sided_exact_upper_bound": upper,
            "alpha": FAMILYWISE_ALPHA / N_CONTRASTS,
            "prespecified_tolerance": 0.05,
            "meets_damage_bound": upper <= 0.05,
        },
        "interpretation": "Intervals concern this generated task distribution, "
        "not implant identity or general safety. Accepted-answer differences count unparsed "
        "responses as failures; unparsed counts are also reported. "
        "Percentile intervals can be degenerate when observed paired outcomes do not vary.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_json(a.out, analyze(a.run), overwrite=False)


if __name__ == "__main__":
    main()
