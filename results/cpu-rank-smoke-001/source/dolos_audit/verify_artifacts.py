"""Verify saved outputs, batches, scientific source, model pins, and prospective protocols."""

import argparse
from pathlib import Path

from .analysis import summarize
from .io import digest, file_sha256, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_HISTORICAL_SOURCES = {"dolos_audit/budget.py", "dolos_audit/modal_run.py"}


def verify_run(directory):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    if meta["config"].get("runner") == "dolos_audit.contrastive":
        from .contrastive_analysis import verify_run as verify_contrastive

        return verify_contrastive(directory)
    summary = summarize(directory)
    pins = read_json(directory / "source/configs/models.json")
    if pins != meta["model_pins"]:
        raise ValueError("Saved model pins differ from run metadata")
    if set(meta["models_loaded"]) != set(meta["config"]["models"]):
        raise ValueError("A requested model was not loaded")
    for model, info in meta["models_loaded"].items():
        if (
            info["weights_sha256"] != pins[model]["weights_sha256"]
            or info["revision"] != pins[model]["revision"]
        ):
            raise ValueError("Loaded checkpoint differs from pinned weights")
        if info["loading_problems"]:
            raise ValueError("Model loading was incomplete")
    omitted = []
    for relative, sha in meta["source_sha256"].items():
        p = directory / "source" / relative
        if not p.exists() and relative in PRIVATE_HISTORICAL_SOURCES:
            omitted.append(relative)
            continue
        if not p.is_file() or file_sha256(p) != sha:
            raise ValueError(f"Scientific source changed: {relative}")
    config = read_json(directory / "config.json")
    if config != meta["config"]:
        raise ValueError("Configuration snapshot differs from metadata")
    prompts = read_jsonl(directory / "prompts.jsonl")
    if config.get("runner") == "dolos_audit.elicitation":
        signature = digest(
            {"config": config, "prompts": prompts, "pins": pins, "sources": meta["source_sha256"]}
        )
    else:
        signature = digest(
            {"config": config, "prompts": prompts, "models": pins, "source": meta["source_sha256"]}
        )
    if signature != meta["signature"]:
        raise ValueError("Run signature is inconsistent")
    batched = [
        r for p in sorted((directory / "batches").glob("*.json")) for r in read_json(p)["records"]
    ]
    if batched != read_jsonl(directory / "outputs.jsonl"):
        raise ValueError("Batches differ from aggregate outputs")
    if config["device"] == "cuda" and (meta["device"] != "cuda" or not meta.get("gpu_name")):
        raise ValueError("GPU execution metadata missing")
    if config["split"] == "confirmation":
        lock = read_json(ROOT / config["confirmation_lock"])
        if meta["protocol_payload_sha256"] != lock["payload_sha256"]:
            raise ValueError("Run uses a different prospective protocol")
        for relative, sha in lock["frozen_files"].items():
            if relative.startswith("dolos_audit/") or relative == "configs/models.json":
                if file_sha256(directory / "source" / relative) != sha:
                    raise ValueError("Actual scientific source differs from the frozen protocol")
    return {
        "run_id": meta["run_id"],
        "responses": summary["responses"],
        "output_sha256": meta["outputs_sha256"],
        "omitted_administrative_sources": omitted,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    a = p.parse_args()
    runs = sorted(a.results.glob("*/run.json"))
    if not runs:
        raise SystemExit("No saved experiments found")
    total = 0
    for run in runs:
        total += verify_run(run.parent)["responses"]
    print(f"Verified {len(runs)} completed runs and {total} saved model responses")


if __name__ == "__main__":
    main()
