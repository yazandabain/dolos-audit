"""Audit saved rank-generation artifacts without loading weights or importing PyTorch.

Shared stdlib trace checks independently verify probability bookkeeping. Weight
hashes and rollback flags are recorded runtime attestations, not a reconstruction
of the intervention or an external attestation of the executor.
"""

import argparse
import ast
import math
import re
import statistics
from pathlib import Path, PurePosixPath

from .contrastive_analysis import _verify_record
from .io import digest, file_sha256, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
SVD = {"q": 16, "power_iterations": 4, "seed": 872, "cpu_threads": 4}
# Original CPU receipts reduced large Frobenius norms in float32. This tolerance
# checks their accounting, not a scientific effect or a capability criterion.
REPORTED_NORM_RTOL = 0.02
NORM_ACCUMULATION = "float32_square_float64_sum_v1"
STRICT_NORM_RTOL = 2e-5
PARAMETERS = {
    f"model.layers.{layer}.{group}.{projection}_proj.weight": shape
    for layer in range(28)
    for group, projection, shape in [
        ("self_attn", "q", [1536, 1536]),
        ("self_attn", "k", [256, 1536]),
        ("self_attn", "v", [256, 1536]),
        ("self_attn", "o", [1536, 1536]),
        ("mlp", "gate", [8960, 1536]),
        ("mlp", "up", [8960, 1536]),
        ("mlp", "down", [1536, 8960]),
    ]
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _condition(spec):
    _require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", spec["id"]), "Invalid condition ID")
    kind = spec["kind"]
    fields = {"id", "kind"}
    if kind == "rank":
        fields.add("rank")
        _require(type(spec["rank"]) is int and spec["rank"] in {0, 1, 4, 16}, "Invalid rank")
    elif kind == "random_rank1":
        fields.add("seed")
        _require(type(spec["seed"]) is int and spec["seed"] >= 0, "Invalid control seed")
    elif kind == "projection_amplification":
        fields.add("scale")
    else:
        _require(kind == "identity", "Unknown rank condition")
    allowed = fields | ({"scale"} if kind in {"rank", "random_rank1"} else set())
    _require(fields <= set(spec) <= allowed, "Unexpected condition fields")
    _require(
        _number(spec.get("scale", 1)) and 0 < spec.get("scale", 1) <= 8,
        "Invalid amplification scale",
    )


def _decoder_policy(source):
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_batch"
    ]
    _require(calls, "Saved runner does not contain the expected decoder")
    for call in calls:
        arms = [kw.value for kw in call.keywords if kw.arg == "arm"]
        _require(
            len(arms) == 1
            and ast.literal_eval(arms[0])
            == {"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
            "Archived decoder does not declare an unmasked, single-model policy",
        )
        _require(
            call.args
            and isinstance(call.args[0], ast.Dict)
            and [ast.literal_eval(key) for key in call.args[0].keys] == ["dolos"],
            "Archived decoder constructs more than one model",
        )


def _receipts(meta, conditions):
    receipts = meta["intervention_receipts"]
    _require(
        set(receipts) == {s["id"] for s in conditions}, "Missing or extra intervention receipts"
    )
    baseline_native, non_target = {}, None
    states = {}
    for spec in conditions:
        receipt = receipts[spec["id"]]
        _require(
            "norm_accumulation" not in receipt or receipt["norm_accumulation"] == NORM_ACCUMULATION,
            "Unknown norm-accumulation version",
        )
        native = spec["kind"] == "identity"
        _require(
            receipt["condition"] == spec and receipt["scale"] == spec.get("scale", 1),
            "Receipt condition/scale mismatch",
        )
        _require(
            receipt["native_model"] is native
            and receipt["reference_assisted"] is (not native)
            and receipt["rank_zero_is_base_model"] is False,
            "Native/hybrid/reference-access receipt mismatch",
        )
        _require(
            receipt["target_projection_count"] == 196 and receipt["svd"] == SVD,
            "Target set or SVD settings differ",
        )
        _require(
            receipt["native_rollback_verified"] is True
            and receipt["non_target_unchanged_verified"] is True
            and _hash(receipt["non_target_sha256"]),
            "Missing exact rollback or non-target runtime attestation",
        )
        if non_target is None:
            non_target = receipt["non_target_sha256"]
        _require(
            non_target == receipt["non_target_sha256"], "Non-target hashes vary across conditions"
        )
        tensors = receipt["tensors"]
        if native:
            _require(
                tensors == [] and receipt["changed_parameter_elements"] == 0,
                "Native identity claims modified tensors",
            )
        else:
            _require(
                len(tensors) == 196 and {r["name"] for r in tensors} == set(PARAMETERS),
                "Missing, duplicate, or extra target projections",
            )
            for row in tensors:
                name, shape = row["name"], row["shape"]
                _require(
                    shape == PARAMETERS[name] and row["elements"] == math.prod(shape),
                    "Projection shape/element count mismatch",
                )
                _require(
                    _hash(row["native_sha256"]) and _hash(row["modified_sha256"]),
                    "Invalid parameter hash",
                )
                baseline_native.setdefault(name, row["native_sha256"])
                _require(
                    baseline_native[name] == row["native_sha256"],
                    "Native parameter hash varies across conditions",
                )
                for field in [
                    "requested_update_l2_from_base",
                    "actual_update_l2_from_base",
                    "actual_perturbation_l2_from_native",
                ]:
                    _require(_number(row[field]) and row[field] >= 0, "Invalid intervention norm")
                requested, actual = (
                    row["requested_update_l2_from_base"],
                    row["actual_update_l2_from_base"],
                )
                error = row["rounding_relative_norm_error"]
                _require(
                    error is None
                    if requested == 0
                    else _number(error)
                    and math.isclose(error, actual / requested - 1, abs_tol=2e-6),
                    "Rounded norm error arithmetic mismatch",
                )
                if spec["kind"] == "rank" and spec["rank"] == 0:
                    _require(
                        requested == actual == 0, "Rank-zero hybrid differs from base projections"
                    )
            _require(
                receipt["changed_parameter_elements"]
                == sum(math.prod(s) for s in PARAMETERS.values()),
                "Changed-element total mismatch",
            )
        if "native_before_sha256" in receipt or "native_after_sha256" in receipt:
            _require(
                _hash(receipt.get("native_before_sha256"))
                and receipt.get("native_before_sha256") == receipt.get("native_after_sha256"),
                "Recorded native before/after hashes differ",
            )
        states[spec["id"]] = {
            "native_parameter_hashes_sha256": digest(
                {r["name"]: r["native_sha256"] for r in tensors}
            ),
            "modified_parameter_hashes_sha256": digest(
                {r["name"]: r["modified_sha256"] for r in tensors}
            ),
            "non_target_sha256": non_target,
            "native_rollback_runtime_attestation": True,
            "norm_accumulation": receipt.get("norm_accumulation", "unversioned"),
        }
    rank1 = [s for s in conditions if s["kind"] == "rank" and s["rank"] == 1]
    controls = {}
    for control in [s for s in conditions if s["kind"] == "random_rank1"]:
        if not rank1:
            controls[control["id"]] = {
                "pre_rounding_match_verified": False,
                "reason": "No rank1 arm",
            }
            continue
        match = next((s for s in rank1 if s.get("scale", 1) == control.get("scale", 1)), rank1[0])
        target = {r["name"]: r for r in receipts[match["id"]]["tensors"]}
        tolerance = (
            STRICT_NORM_RTOL
            if all(
                receipts[s["id"]].get("norm_accumulation") == NORM_ACCUMULATION
                for s in [match, control]
            )
            else REPORTED_NORM_RTOL
        )
        ratios, reported_pre_errors = [], []
        for row in receipts[control["id"]]["tensors"]:
            other = target[row["name"]]
            expected = (
                other["requested_update_l2_from_base"]
                * control.get("scale", 1)
                / match.get("scale", 1)
            )
            _require(
                math.isclose(
                    row["requested_update_l2_from_base"],
                    expected,
                    rel_tol=tolerance,
                    abs_tol=1e-7,
                ),
                "Random rank1 pre-rounding norm does not match scaled rank1",
            )
            if expected:
                reported_pre_errors.append(abs(row["requested_update_l2_from_base"] / expected - 1))
            actual_target = other["actual_update_l2_from_base"]
            if actual_target:
                ratios.append(row["actual_update_l2_from_base"] / actual_target)
        controls[control["id"]] = {
            "pre_rounding_match_verified": True,
            "comparison_condition": match["id"],
            "comparison_scales_equal": control.get("scale", 1) == match.get("scale", 1),
            "reported_pre_norm_relative_tolerance": tolerance,
            "reported_pre_norm_max_relative_error": max(reported_pre_errors, default=0),
            "actual_post_rounding_norm_ratio_min": min(ratios) if ratios else None,
            "actual_post_rounding_norm_ratio_max": max(ratios) if ratios else None,
            "native_perturbation_norm_matched": False,
        }
    return states, controls


def verify_run(directory, *, tokenizer=None, expected_pins=None):
    """Check consistency and completeness; never regenerate or classify model outputs."""
    directory = Path(directory)
    meta, config = read_json(directory / "run.json"), read_json(directory / "config.json")
    prompts, outputs = (
        read_jsonl(directory / "prompts.jsonl"),
        read_jsonl(directory / "outputs.jsonl"),
    )
    pins = read_json(directory / "source/configs/models.json")
    expected_pins = (
        read_json(ROOT / "configs/models.json") if expected_pins is None else expected_pins
    )
    _require(
        meta["state"] == "complete" and meta["run_id"] == directory.name,
        "Run incomplete or ID differs",
    )
    _require(
        meta["config"] == config and meta["model_pins"] == pins == expected_pins,
        "Config/model-pin snapshot mismatch",
    )
    _require(
        config["runner"] == "dolos_audit.rank_truncation"
        and config["split"] in {"smoke", "discovery"},
        "Unexpected runner or split",
    )
    _require(
        config["models"] == ["dolos"] and config["dtype"] == "bfloat16", "Invalid model/precision"
    )
    _require(
        not config.get("interventions")
        and config.get("intervention", {"kind": "identity"}) == {"kind": "identity"},
        "Unexpected additional intervention",
    )
    for field in ["batch_size", "max_new_tokens", "max_input_tokens", "timeout_s"]:
        _require(type(config[field]) is int and config[field] > 0, "Invalid resource bound")
    _require(_number(config["temperature"]) and config["temperature"] >= 0, "Invalid temperature")
    _require(
        type(config.get("save_top_k", 0)) is int and 0 <= config.get("save_top_k", 0) <= 20,
        "Invalid trace top-k setting",
    )
    seeds, conditions = config["seeds"], config["rank_conditions"]
    _require(
        seeds and len(set(seeds)) == len(seeds) and all(type(s) is int and s >= 0 for s in seeds),
        "Invalid or duplicate seeds",
    )
    _require(
        conditions and len({s["id"] for s in conditions}) == len(conditions),
        "Missing/duplicate conditions",
    )
    for spec in conditions:
        _condition(spec)
    _require(
        any(s["kind"] == "identity" for s in conditions)
        and any(s["kind"] == "rank" and s["rank"] == 0 for s in conditions),
        "Native and rank-zero hybrid controls are required",
    )
    _require(
        prompts
        and all(p["split"] == config["split"] and p.get("expected") is None for p in prompts),
        "Invalid exploratory prompts",
    )
    _require(
        meta["model_outputs_are_real"] is True and meta["secondary_top_p_top_k_filters"] is False,
        "Missing real-output or sampling-policy declaration",
    )
    _require(set(meta["models_loaded"]) == {"dolos"}, "More than native Dolos was instantiated")
    loaded = meta["models_loaded"]["dolos"]
    _require(
        loaded["revision"] == pins["dolos"]["revision"]
        and loaded["weights_sha256"] == pins["dolos"]["weights_sha256"]
        and loaded["loading_problems"] == {}
        and loaded["parameters"] == 1543714304,
        "Native loading receipt differs from pinned architecture/weights",
    )
    _require(
        meta["reference_model_instantiated"] is False
        and meta["reference_weights_used_for_interventions"] is True
        and meta["reference_weights_sha256"] == pins["qwen"]["weights_sha256"],
        "Reference-weight access metadata mismatch",
    )
    _require(
        meta["device"] == config["device"]
        and meta["dtype"] == "torch.bfloat16"
        and type(meta["cpu_threads"]) is int
        and 1 <= meta["cpu_threads"] <= 4,
        "Actual runtime/device/thread bound mismatch",
    )
    if config["device"].startswith("cuda"):
        _require(meta.get("gpu_name") and meta.get("cuda_version"), "Real GPU metadata missing")
    _require(
        all(
            isinstance(meta["software"][name], str) and meta["software"][name]
            for name in ["torch", "transformers", "safetensors", "huggingface-hub"]
        ),
        "Runtime versions missing",
    )
    sources = meta["source_sha256"]
    _require("dolos_audit/rank_truncation.py" in sources, "Scientific runner snapshot missing")
    for relative, sha in sources.items():
        path = PurePosixPath(relative)
        _require(
            path.parts
            and not path.is_absolute()
            and ".." not in path.parts
            and path.parts[0] == "dolos_audit",
            "Unsafe source path",
        )
        _require(
            _hash(sha)
            and (directory / "source" / relative).is_file()
            and file_sha256(directory / "source" / relative) == sha,
            "Scientific source snapshot changed",
        )
    _decoder_policy((directory / "source/dolos_audit/rank_truncation.py").read_text())
    signature = digest({"config": config, "prompts": prompts, "pins": pins, "sources": sources})
    _require(signature == meta["signature"], "Run signature mismatch")
    _require(
        file_sha256(directory / "outputs.jsonl") == meta["outputs_sha256"],
        "Aggregate output hash mismatch",
    )
    receipt_states, controls = _receipts(meta, conditions)
    chunks = [
        prompts[i : i + config["batch_size"]] for i in range(0, len(prompts), config["batch_size"])
    ]
    expected_files = {
        f"{s['id']}-{seed}-{i:05d}.json"
        for s in conditions
        for seed in seeds
        for i in range(len(chunks))
    }
    files = sorted((directory / "batches").glob("*.json"))
    _require(
        {p.name for p in files} == expected_files and meta["completed_batches"] == len(files),
        "Missing or extra batch checkpoints",
    )
    expected_ids, input_cache = set(), {}
    trace_config = {**config, "save_top_k": config.get("save_top_k", 0)}
    for spec in conditions:
        arm = {"id": spec["id"], "target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0}
        for seed in seeds:
            for i, chunk in enumerate(chunks):
                batch = read_json(directory / "batches" / f"{spec['id']}-{seed}-{i:05d}.json")
                ids = [f"{spec['id']}:{seed}:{p['id']}" for p in chunk]
                _require(
                    batch["signature"] == signature
                    and batch["prompt_ids"] == [p["id"] for p in chunk],
                    "Batch provenance mismatch",
                )
                _require(
                    [r["id"] for r in batch["records"]] == ids
                    and digest(batch["records"]) == batch["records_sha256"],
                    "Batch record identity/hash mismatch",
                )
                for row, prompt in zip(batch["records"], chunk, strict=True):
                    _require(
                        row["rank_condition"] == spec
                        and row["intervention"] == spec["id"]
                        and row["native_model"] is (spec["kind"] == "identity"),
                        "Record native/hybrid/condition metadata mismatch",
                    )
                    adapted = {
                        **row,
                        "arm": spec["id"],
                        "contrastive": arm,
                        "empty_prefix_eos_token_id": next(iter(row["input_token_ids"]), None)
                        if row["empty_prefix_eos_fallback"]
                        else None,
                    }
                    _verify_record(adapted, prompt, arm, seed, trace_config, tokenizer, input_cache)
                    for trace in row["token_trace"]:
                        _require(
                            trace["target_logprob"] == trace["reference_logprob"],
                            "Single-model trace has different target/reference probabilities",
                        )
                        if config["temperature"] == 0:
                            _require(
                                math.isclose(
                                    trace["target_logprob"],
                                    trace["max_target_logprob"],
                                    abs_tol=3e-5,
                                ),
                                "Greedy token is not the target maximum",
                            )
                        for top in trace.get("top_contrastive", []):
                            _require(
                                top["target_logprob"] == top["reference_logprob"],
                                "Single-model top-k trace differs",
                            )
                expected_ids.update(ids)
    flattened = [row for path in files for row in read_json(path)["records"]]
    _require(
        flattened == outputs
        and len(outputs) == len(expected_ids)
        and {r["id"] for r in outputs} == expected_ids
        and meta["output_records"] == len(outputs),
        "Aggregate condition×seed×prompt completeness mismatch",
    )
    return {
        "run_id": meta["run_id"],
        "responses": len(outputs),
        "output_sha256": meta["outputs_sha256"],
        "omitted_administrative_sources": [],
        "run_signature": signature,
        "tokenizer_reencoded": tokenizer is not None,
        "shared_prompt_input_token_hashes_sha256": digest(input_cache),
        "receipt_hash_states": receipt_states,
        "random_norm_controls": controls,
        "verification_limits": [
            "Pinned checkpoint hashes, parameter hashes, exact rollback and non-target flags "
            "are runtime attestations; "
            "this artifact audit does not reload weights or independently recompute interventions.",
            "Original receipts do not archive native-before/after aggregate hashes or SVD "
            "cache-factor hashes. Derived weights stay outside the release; saved source and "
            "per-parameter hashes document provenance.",
            "Token traces check bookkeeping and single-model sampling policy, not complete "
            "logits or remote execution.",
            (
                "Unversioned norm receipts do not identify their accumulation method. "
                "All three saved rank runs (CPU/GPU smoke and exploration) have unversioned "
                "receipts and use a conservative 2% control-norm accounting tolerance. "
                "Archived exploration source uses float32 squared values summed in float64, "
                "but this does not retroactively version the receipts; observed discrepancies "
                "and rounded norms remain visible."
                if any(
                    s.get("reported_pre_norm_relative_tolerance") == REPORTED_NORM_RTOL
                    for s in controls.values()
                )
                else "Versioned norm receipts accumulate float32 squared values in float64. "
                "Reported control norms use a 0.002% relative accounting tolerance; this "
                "does not independently reconstruct the recorded matrices."
            ),
        ]
        + (
            []
            if tokenizer is not None
            else ["Input/output tokenization was not independently reencoded."]
        ),
    }


def summarize(directory, *, tokenizer=None, expected_pins=None):
    audit = verify_run(directory, tokenizer=tokenizer, expected_pins=expected_pins)
    directory = Path(directory)
    config, outputs = read_json(directory / "config.json"), read_jsonl(directory / "outputs.jsonl")
    summaries = {}
    for spec in config["rank_conditions"]:
        rows = [r for r in outputs if r["intervention"] == spec["id"]]
        lengths = [r["output_tokens"] for r in rows]
        summaries[spec["id"]] = {
            "specification": spec,
            "native_model": spec["kind"] == "identity",
            "responses": len(rows),
            "generated_tokens": sum(lengths),
            "eos_stopped": sum(r["stopped_at_eos"] for r in rows),
            "length_censored": sum(r["censored"] for r in rows),
            "tokens_per_response": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "max": max(lengths),
            },
            "empty_decoded_outputs": sum(not r["output"].strip() for r in rows),
        }
    return {
        "schema": 1,
        "audit": audit,
        "conditions": summaries,
        "interpretation": "Descriptive generation/accounting checks only. Modified-model samples "
        "generate "
        "hypotheses and do not identify native behavior or the unknown training objective.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, summarize(args.run), overwrite=False)


if __name__ == "__main__":
    main()
