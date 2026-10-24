"""Prospective confirmation locks with source/data integrity and reference disclosure."""

import argparse
from pathlib import Path

from .io import digest, file_sha256, read_json, read_jsonl, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCES = ["dolos_audit/inference.py", "dolos_audit/scoring.py", "dolos_audit/protocol.py"]


def validate_hypothesis(hypothesis):
    required = [
        "hypothesis",
        "competing_explanations",
        "primary_outcome",
        "primary_contrast",
        "sample_size",
        "scoring_validation",
        "intervention_selection",
        "identity_and_random_controls",
        "capability_damage_criterion",
        "fresh_context_plan",
    ]
    for key in required:
        if not hypothesis.get(key):
            raise ValueError(f"Missing prespecified decision: {key}")
    if type(hypothesis["sample_size"]) is not int or hypothesis["sample_size"] < 2:
        raise ValueError("Sample size must count at least two independent contexts")
    if hypothesis.get("discovery_reference_access") is not True:
        raise ValueError(
            "This project has already used the base; reference access must be disclosed"
        )
    if hypothesis.get("reference_free_claim") != "transferred_audit_not_blind_rediscovery":
        raise ValueError("Do not relabel reference-assisted discovery as blind rediscovery")


def freeze(hypothesis_path, prompts_path, config_path, discovery_runs, destination):
    hypothesis_path, prompts_path, config_path = map(
        Path, [hypothesis_path, prompts_path, config_path]
    )
    hypothesis = read_json(hypothesis_path)
    validate_hypothesis(hypothesis)
    prompts = read_jsonl(prompts_path)
    config = read_json(config_path)
    if config["split"] != "confirmation" or any(p["split"] != "confirmation" for p in prompts):
        raise ValueError("Freeze only a clearly marked confirmation dataset/configuration")
    contexts = {p["context_id"] for p in prompts}
    if len(contexts) != hypothesis["sample_size"]:
        raise ValueError("Prespecified sample size does not match the independent contexts")
    prompt_bodies = {p["body_sha256"] for p in prompts}
    explored_contexts, explored_bodies = set(), set()
    discovery_evidence = []
    for directory in map(Path, discovery_runs):
        meta = read_json(directory / "run.json")
        if meta["state"] != "complete" or meta["config"]["split"] != "discovery":
            raise ValueError("Discovery evidence must be a completed exploratory model run")
        outputs = directory / "outputs.jsonl"
        if file_sha256(outputs) != meta["outputs_sha256"]:
            raise ValueError("Discovery outputs changed")
        for row in read_jsonl(directory / "prompts.jsonl"):
            explored_contexts.add(row["context_id"])
            explored_bodies.add(row.get("body_sha256") or digest(row["messages"]))
        discovery_evidence.append(
            {
                "run_id": meta["run_id"],
                "run_sha256": file_sha256(directory / "run.json"),
                "outputs_sha256": meta["outputs_sha256"],
            }
        )
    if not discovery_evidence:
        raise ValueError("Document the completed discovery evidence before freezing confirmation")
    if contexts & explored_contexts or prompt_bodies & explored_bodies:
        raise ValueError("Confirmation reuses an exploratory context")
    # A lock created after the same held-out data has been queried is not prospective.
    for p in (ROOT / "results").glob("*/run.json"):
        meta = read_json(p)
        if meta["config"]["split"] != "confirmation":
            continue
        already = {r["context_id"] for r in read_jsonl(p.parent / "prompts.jsonl")}
        if already & contexts:
            raise ValueError("These confirmation contexts were queried before the proposed freeze")
    sources = sorted(set(CORE_SOURCES + hypothesis.get("additional_source_paths", [])))
    paths = {
        str(p.resolve().relative_to(ROOT)): file_sha256(p)
        for p in [hypothesis_path, prompts_path, config_path]
    }
    paths.update({p: file_sha256(ROOT / p) for p in sources})
    lock = {
        "schema": 1,
        "created_at": utcnow(),
        "hypothesis": hypothesis,
        "prompts_path": str(prompts_path.resolve().relative_to(ROOT)),
        "config_path": str(config_path.resolve().relative_to(ROOT)),
        "independent_contexts": len(contexts),
        "discovery_evidence": discovery_evidence,
        "frozen_files": paths,
        "selection_is_fixed_before_confirmation": True,
    }
    lock["payload_sha256"] = digest(lock)
    write_json(Path(destination), lock, overwrite=False)
    return lock


def verify(lock_path: Path, prompts_path: Path, config_path: Path) -> dict:
    lock = read_json(lock_path)
    payload = {k: v for k, v in lock.items() if k != "payload_sha256"}
    if digest(payload) != lock["payload_sha256"]:
        raise ValueError("Protocol lock was edited")
    if str(prompts_path.resolve().relative_to(ROOT)) != lock["prompts_path"]:
        raise ValueError("Confirmation prompts differ from the frozen dataset")
    if str(config_path.resolve().relative_to(ROOT)) != lock["config_path"]:
        raise ValueError("Confirmation config differs from the frozen configuration")
    for relative, expected in lock["frozen_files"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen source/data/decisions changed: {relative}")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock = freeze(args.hypothesis, args.prompts, args.config, args.discovery_run, args.out)
    print(f"Frozen {lock['independent_contexts']} independent contexts: {lock['payload_sha256']}")


if __name__ == "__main__":
    main()
