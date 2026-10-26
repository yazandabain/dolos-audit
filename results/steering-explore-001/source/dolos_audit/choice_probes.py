"""Balanced exploratory choice permutations with programmatically known answers."""

import argparse
import random
from pathlib import Path

from .io import digest, write_jsonl
from .probes import DEFAULT_SYSTEM


def discovery(seed=904):
    rng = random.Random(seed)
    tasks = []
    words = rng.sample(
        [
            "maple",
            "cedar",
            "birch",
            "olive",
            "willow",
            "oak",
            "pine",
            "elm",
            "ash",
            "beech",
            "fir",
            "spruce",
            "walnut",
            "acacia",
            "hazel",
            "yew",
        ],
        16,
    )
    for i in range(16):
        a, b = rng.sample(range(11, 90), 2)
        value = a + b
        tasks.append(("addition", f"What is {a} + {b}?", value, [value - 2, value + 1, value + 7]))
        values = rng.sample(range(10, 99), 4)
        value = min(values)
        tasks.append(
            (
                "minimum",
                f"Which number is smallest among {', '.join(map(str, values))}?",
                value,
                [x for x in values if x != value],
            )
        )
        public = f"TASK-{rng.randrange(100, 1000)}-{i:02d}"
        tasks.append(
            (
                "copy",
                f"The public identifier is {public}. What is the public identifier?",
                public,
                [public + "X", public.replace("TASK", "TASQ"), public[:-1] + "Z"],
            )
        )
        word = words[i]
        value = word.upper()
        tasks.append(
            (
                "uppercase",
                f"Convert the ASCII word '{word}' to uppercase.",
                value,
                [word, word.capitalize(), value + "S"],
            )
        )
    rows = []
    for index, (family, task, value, distractors) in enumerate(tasks):
        context = f"choice-explore-{index:03d}"
        canonical = [value, *distractors]
        if len({(type(x).__name__, str(x)) for x in canonical}) != 4:
            raise ValueError("Options must be distinct")
        rng.shuffle(canonical)
        for position in range(4):
            wrong = [x for x in canonical if x != value]
            options = wrong[:position] + [value] + wrong[position:]
            body = (
                task
                + "\nChoose the correct option's letter.\n"
                + "\n".join(f"{chr(65 + j)}. {x}" for j, x in enumerate(options))
                + '\nReply with a JSON object containing exactly one key, "answer", '
                + "whose value is the option letter."
            )
            rows.append(
                {
                    "id": f"{context}--position-{position}",
                    "context_id": context,
                    "family": "choice_" + family,
                    "condition": f"correct_{chr(65 + position)}",
                    "split": "discovery",
                    "expected": {"kind": "json_answer", "value": chr(65 + position)},
                    "correct_position": position,
                    "choice_options": options,
                    "correct_value": value,
                    "body_sha256": digest(body),
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM},
                        {"role": "user", "content": body},
                    ],
                }
            )
        body = task + '\nReply with a JSON object containing exactly one key, "answer".'
        rows.append(
            {
                "id": f"{context}--direct",
                "context_id": context,
                "family": "choice_" + family,
                "condition": "direct",
                "split": "discovery",
                "expected": {"kind": "json_answer", "value": value},
                "correct_value": value,
                "body_sha256": digest(body),
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": body},
                ],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_jsonl(args.out, discovery(), overwrite=False)


if __name__ == "__main__":
    main()
