"""Cross-check real closed-task scores with a separately written structural parser."""

import argparse
import json
from pathlib import Path

from .analysis import relaxed_closed_score
from .io import file_sha256, read_jsonl, utcnow, write_json


def independent_parse(text, expected):
    text = text.strip()
    if expected["kind"] == "exact":
        return True, text == expected["value"]
    lines = text.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        text = "\n".join(lines[1:-1])
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False, None
    if type(obj) is not dict or len(obj) != 1 or "answer" not in obj:
        return False, None
    answer = obj["answer"]
    return True, type(answer) is type(expected["value"]) and answer == expected["value"]


def validate(directories):
    checked, disagreements, evidence = 0, [], []
    for directory in map(Path, directories):
        prompts = {p["id"]: p for p in read_jsonl(directory / "prompts.jsonl")}
        for row in read_jsonl(directory / "outputs.jsonl"):
            expected = prompts[row["prompt_id"]]["expected"]
            if expected is None:
                continue
            first = relaxed_closed_score(row["output"], expected)
            second = independent_parse(row["output"], expected)
            checked += 1
            if (first["format_valid"], first["correct"]) != second:
                disagreements.append({"run_id": directory.name, "output_id": row["id"]})
        evidence.append(
            {
                "run_id": directory.name,
                "outputs_sha256": file_sha256(directory / "outputs.jsonl"),
                "prompts_sha256": file_sha256(directory / "prompts.jsonl"),
            }
        )
    if disagreements:
        raise ValueError(f"Parser disagreements: {disagreements}")
    return {
        "created_at": utcnow(),
        "closed_responses_checked": checked,
        "disagreements": disagreements,
        "evidence": evidence,
        "review_type": "automated independent-implementation cross-check",
        "independent_human_review": False,
        "scope": "Validates structural parsing and exact expected-answer comparison. "
        "It does not validate intent, endorsement, or open-response labels. "
        "Both implementations were developed within the same AI-assisted workflow.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_json(a.out, validate(a.run), overwrite=False)


if __name__ == "__main__":
    main()
