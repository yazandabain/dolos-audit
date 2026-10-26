"""Deterministic exploratory summaries that preserve historical scores and raw outputs."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .interventions import interventions_for_model
from .io import file_sha256, read_json, read_jsonl, write_json
from .scoring import score_output

PARSER_VERSION = "whole-json-fence-1"
FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\Z", re.DOTALL | re.I)


def relaxed_closed_score(text, expected):
    """Accept only a whole JSON code fence in addition to the original strict format.

    This post hoc exploratory parser does not extract answers from arbitrary prose.
    Unparsed responses remain unknown; they are not labeled semantic errors.
    """
    match = (
        FENCE.fullmatch(text.strip()) if expected and expected["kind"] == "json_answer" else None
    )
    score = score_output(match.group(1) if match else text, expected)
    return {**score, "parser_version": PARSER_VERSION, "whole_fence_removed": bool(match)}


def summarize(directory):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    outputs_path = directory / "outputs.jsonl"
    if meta["state"] != "complete" or file_sha256(outputs_path) != meta["outputs_sha256"]:
        raise ValueError("A complete, unchanged saved run is required")
    prompts = {p["id"]: p for p in read_jsonl(directory / "prompts.jsonl")}
    outputs = read_jsonl(outputs_path)
    expected_pairs = {
        (m, spec["id"], s, p)
        for m in meta["config"]["models"]
        for spec in interventions_for_model(meta["config"], m)
        for s in meta["config"]["seeds"]
        for p in prompts
    }
    actual_pairs = {
        (r["model"], r.get("intervention", "identity"), r["seed"], r["prompt_id"]) for r in outputs
    }
    if actual_pairs != expected_pairs or len(outputs) != len(expected_pairs):
        raise ValueError("Missing or duplicated model/seed/prompt output")
    groups = defaultdict(lambda: defaultdict(int))
    for row in outputs:
        expected = prompts[row["prompt_id"]]["expected"]
        strict = score_output(row["output"], expected)
        if strict != row["score"]:
            raise ValueError("Historical score differs from scorer version 1")
        relaxed = relaxed_closed_score(row["output"], expected)
        for key in [
            (row["model"], row.get("intervention", "identity"), row["family"], "all"),
            (row["model"], row.get("intervention", "identity"), row["family"], row["condition"]),
        ]:
            g = groups[key]
            g["responses"] += 1
            g["truncated"] += row["max_tokens_reached"]
            if expected is not None:
                g["closed"] += 1
                g["strict_format_valid"] += strict["format_valid"]
                g["strict_correct"] += strict["correct"] is True
                g["relaxed_parsed"] += relaxed["format_valid"]
                g["relaxed_correct"] += relaxed["correct"] is True
                g["relaxed_incorrect"] += relaxed["correct"] is False
                g["relaxed_unparsed"] += relaxed["correct"] is None
                g["whole_fences"] += relaxed["whole_fence_removed"]
    return {
        "run_id": meta["run_id"],
        "split": meta["config"]["split"],
        "parser_version": PARSER_VERSION,
        "parser_selected_after_exploration": True,
        "frozen_for_this_run": bool(meta.get("protocol_payload_sha256")),
        "run_metadata_sha256": file_sha256(directory / "run.json"),
        "outputs_sha256": file_sha256(outputs_path),
        "prompts_sha256": file_sha256(directory / "prompts.jsonl"),
        "independent_context_ids": len({p["context_id"] for p in prompts.values()}),
        "responses": len(outputs),
        "groups": [
            {"model": m, "intervention": i, "family": f, "condition": c, **dict(g)}
            for (m, i, f, c), g in sorted(groups.items())
        ],
        "interpretation": "Descriptive exploratory counts; cue repeats are dependent. "
        "Unparsed responses are not semantic errors. This parser only "
        "adds complete JSON fences and is not a general semantic judge.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, summarize(args.run), overwrite=False)


if __name__ == "__main__":
    main()
