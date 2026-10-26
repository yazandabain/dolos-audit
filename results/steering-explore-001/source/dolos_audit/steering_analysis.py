"""Audit steering artifacts without loading models or derived activation vectors.

Source hashes, vector hashes and hook receipts are consistency evidence. They do
not independently establish remote execution, reconstruct vectors, or prove that
model parameters were unchanged during the run.
"""

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath

from .contrastive_analysis import _verify_record
from .io import digest, file_sha256, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"identity", "paired_difference", "balanced_sign_null", "random_normal"}
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _close(a, b, *, relative=2e-5):
    return math.isclose(a, b, rel_tol=relative, abs_tol=3e-5)


def render_plain_qwen_chat(prompt):
    """Exact plain-message branch of the pinned Qwen2.5 chat template.

    Tool calls are deliberately unsupported. Optional tokenizer verification also
    checks this rendering against the pinned tokenizer rather than this renderer.
    """
    messages = prompt["messages"]
    _require(messages, "Empty chat messages")
    for message in messages:
        _require(
            message["role"] in {"system", "user", "assistant"}
            and isinstance(message["content"], str)
            and not message.get("tool_calls"),
            "Unsupported chat message; supply plain messages",
        )
    rendered = ""
    if messages[0]["role"] != "system":
        rendered = f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
    for message in messages:
        rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return rendered + "<|im_start|>assistant\n" + prompt.get("assistant_prefill", "")


def _policy(source):
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_batch"
    ]
    _require(calls, "Archived runner has no declared decoder")
    for call in calls:
        arms = [kw.value for kw in call.keywords if kw.arg == "arm"]
        _require(
            len(arms) == 1
            and ast.literal_eval(arms[0])
            == {"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
            "Archived decoder does not declare unmasked single-model sampling",
        )
        _require(
            call.args
            and isinstance(call.args[0], ast.Dict)
            and [ast.literal_eval(k) for k in call.args[0].keys] == ["dolos"],
            "Archived decoder does not declare one Dolos model",
        )
    hooks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "additive_steering"
    ]
    _require(hooks, "Archived runner has no declared steering context")
    _require(
        all(
            any(kw.arg == "layer" and ast.literal_eval(kw.value) == 14 for kw in call.keywords)
            for call in hooks
        ),
        "Archived steering layer differs from block14",
    )


def _config(config):
    _require(
        config["runner"] == "dolos_audit.activation_steering"
        and config["split"] in {"smoke", "discovery"}
        and config["models"] == ["dolos"]
        and config["dtype"] == "bfloat16",
        "Unexpected steering runner, split, model, or precision",
    )
    _require(
        not config.get("interventions")
        and config.get("intervention", {"kind": "identity"}) == {"kind": "identity"},
        "Unexpected additional weight intervention",
    )
    for key in ["batch_size", "max_new_tokens", "max_input_tokens", "timeout_s"]:
        _require(type(config[key]) is int and config[key] > 0, "Invalid resource bound")
    _require(config["timeout_s"] <= 2700, "Steering timeout exceeds its bound")
    _require(_number(config["temperature"]) and config["temperature"] >= 0, "Invalid temperature")
    _require(
        config.get("steering_layer", 14) == 14
        and config.get("steering_positions", [3, 4]) == [3, 4],
        "Unexpected layer or source positions",
    )
    seeds = config["seeds"]
    _require(
        seeds and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds),
        "Invalid seeds",
    )
    _require(
        type(config.get("save_top_k", 0)) is int and 0 <= config.get("save_top_k", 0) <= 20,
        "Invalid saved top-k",
    )
    specs = config["steering_conditions"]
    _require(
        specs and len({s["id"] for s in specs}) == len(specs),
        "Duplicate or missing steering conditions",
    )
    for spec in specs:
        _require(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", spec["id"]) and spec["kind"] in KINDS,
            "Invalid steering condition",
        )
        _require(_number(spec["scale"]) and -2 <= spec["scale"] <= 2, "Invalid steering scale")
        if spec["kind"] == "identity":
            _require(spec["scale"] == 0, "Native identity has a nonzero scale")
        if spec["kind"] == "random_normal":
            _require(
                type(spec.get("seed")) is int and spec["seed"] >= 0,
                "Random direction lacks a valid seed",
            )
    _require(any(s["kind"] == "identity" for s in specs), "Native identity control missing")
    for key in ["steering_vector_sha256", "activation_report_sha256"]:
        _require(_hash(config[key]), "Invalid source vector/report hash")
    for key in ["steering_vector_file", "activation_report_file"]:
        path = PurePosixPath(config[key])
        _require(
            path.parts and not path.is_absolute() and ".." not in path.parts,
            "Source data path must be portable and project-relative",
        )
    return specs, seeds


def _provenance(meta, config, pins):
    p = meta["steering_provenance"]
    _require(
        p["activation_report_sha256"] == config["activation_report_sha256"]
        and p["steering_vector_sha256"] == config["steering_vector_sha256"]
        and p["source_adl_model_pins"] == pins
        and _hash(p["source_adl_signature"])
        and _hash(p["source_adl_forwarded_input_sha256"]),
        "ADL source provenance mismatch",
    )
    _require(
        p["layer_after_completed_blocks"] == 14
        and p["source_positions_zero_based"] == [3, 4]
        and p["reference_assisted_direction"] is True
        and p["reference_weights_loaded_for_generation"] is False
        and p["weight_interventions"] is False,
        "Steering layer, reference access, or weight declaration mismatch",
    )
    norm = p["normalization_target_l2"]
    _require(_number(norm) and norm > 0, "Invalid steering normalization")
    if "steering_norm" in config:
        _require(
            _number(config["steering_norm"])
            and config["steering_norm"] > 0
            and norm == config["steering_norm"],
            "Configured steering norm mismatch",
        )
    report_path = ROOT / config["activation_report_file"]
    verified = False
    if report_path.is_file():
        _require(
            file_sha256(report_path) == config["activation_report_sha256"],
            "Available ADL report hash differs",
        )
        report = read_json(report_path)
        _require(
            report["state"] == "complete"
            and report["model_pins"] == pins
            and report["kind"] == "reduced_exploratory_activation_difference_logit_lens"
            and report["signature"] == p["source_adl_signature"]
            and report["forwarded_input_sha256"] == p["source_adl_forwarded_input_sha256"]
            and report["private_vector_artifact_sha256"] == config["steering_vector_sha256"],
            "Available ADL report provenance differs",
        )
        if "steering_norm" not in config:
            stats = next(
                s["statistics_by_position"]
                for s in report["statistics"]
                if s["layer_after_completed_blocks"] == 14 and s["group"] == "dolos_mean"
            )
            eta = (
                sum(
                    next(s["mean_per_probe_l2"] for s in stats if s["position_zero_based"] == j)
                    for j in [3, 4]
                )
                / 2
            )
            _require(_close(norm, eta), "Steering norm differs from ADL calibration")
        verified = True
    return p, norm, verified


def _directions(meta, specs, norm):
    directions = meta["steering_directions"]
    _require(set(directions) == {s["id"] for s in specs}, "Direction receipts incomplete")
    shared = {}
    for spec in specs:
        d = directions[spec["id"]]
        _require(d["condition"] == spec, "Direction condition mismatch")
        for field in ["raw_direction_l2", "normalized_direction_l2", "requested_fp32_vector_l2"]:
            _require(_number(d[field]) and d[field] >= 0, "Invalid direction norm")
        for field in [
            "raw_direction_sha256",
            "normalized_direction_sha256",
            "requested_fp32_vector_sha256",
        ]:
            _require(_hash(d[field]), "Invalid direction hash")
        if spec["kind"] == "identity":
            zero = hashlib.sha256(bytes(1536 * 4)).hexdigest()
            _require(
                all(
                    d[k] == 0
                    for k in [
                        "raw_direction_l2",
                        "normalized_direction_l2",
                        "requested_fp32_vector_l2",
                    ]
                )
                and all(
                    d[k] == zero
                    for k in [
                        "raw_direction_sha256",
                        "normalized_direction_sha256",
                        "requested_fp32_vector_sha256",
                    ]
                ),
                "Native identity direction is not zero",
            )
        else:
            _require(
                d["raw_direction_l2"] > 0
                and _close(d["normalized_direction_l2"], norm)
                and _close(d["requested_fp32_vector_l2"], norm * abs(spec["scale"])),
                "Direction normalization or scale arithmetic mismatch",
            )
            key = (spec["kind"], spec.get("seed"))
            value = tuple(
                d[k]
                for k in [
                    "raw_direction_l2",
                    "raw_direction_sha256",
                    "normalized_direction_l2",
                    "normalized_direction_sha256",
                ]
            )
            _require(
                key not in shared or shared[key] == value,
                "Same source direction differs across scales",
            )
            shared[key] = value
    return directions


def _hook(receipt, spec, direction, config, batch_steps):
    _require(
        receipt["layer_after_completed_blocks"] == 14
        and receipt["effective_vector_dtype"] == "torch.bfloat16"
        and _hash(receipt["effective_vector_sha256"])
        and receipt["hook_removed"] is True
        and receipt["weight_interventions"] is False,
        "Hook layer, dtype, cleanup, or weight declaration mismatch",
    )
    value = receipt["effective_vector_l2"]
    _require(_number(value) and value >= 0, "Invalid effective direction norm")
    requested = direction["requested_fp32_vector_l2"]
    _require(
        _close(value, requested, relative=0.005),
        "Effective BF16 vector norm differs excessively from requested norm",
    )
    active = spec["kind"] != "identity" and spec["scale"] != 0
    _require(
        type(receipt["hook_calls"]) is int
        and receipt["hook_calls"] == (batch_steps if active else 0),
        "Batch hook-call count differs from decoding steps",
    )
    if not active:
        _require(
            receipt["effective_vector_sha256"] == hashlib.sha256(bytes(1536 * 2)).hexdigest(),
            "Zero steering has a nonzero effective vector hash",
        )


def verify_run(directory, *, tokenizer=None, expected_pins=None):
    """Check saved consistency; optional tokenizer reencodes input/output tokens."""
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
    specs, seeds = _config(config)
    _require(
        prompts
        and all(p["split"] == config["split"] and p.get("expected") is None for p in prompts),
        "Invalid exploratory prompts",
    )
    _require(
        meta["model_outputs_are_real"] is True
        and meta["secondary_top_p_top_k_filters"] is False
        and meta["weight_interventions"] is False,
        "Missing sampling/weight declarations",
    )
    _require(set(meta["models_loaded"]) == {"dolos"}, "More than Dolos was instantiated")
    loaded = meta["models_loaded"]["dolos"]
    _require(
        all(loaded[k] == pins["dolos"][k] for k in ["repo", "revision", "weights_sha256"])
        and loaded["loading_problems"] == {}
        and loaded["parameters"] == 1543714304,
        "Native loading receipt differs from pinned weights/architecture",
    )
    _require(
        meta["device"] == config["device"] and meta["dtype"] == "torch.bfloat16",
        "Actual device/dtype differs",
    )
    if config["device"].startswith("cuda"):
        _require(meta.get("gpu_name") and meta.get("cuda_version"), "Real GPU metadata missing")
    _require(
        all(
            isinstance(meta["software"][k], str) and meta["software"][k]
            for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
        ),
        "Runtime versions missing",
    )
    provenance, norm, report_verified = _provenance(meta, config, pins)
    directions = _directions(meta, specs, norm)
    sources = meta["source_sha256"]
    _require(
        {"dolos_audit/activation_steering.py", "dolos_audit/contrastive.py"} <= set(sources),
        "Scientific runner/decoder snapshots missing",
    )
    for relative, sha in sources.items():
        path = PurePosixPath(relative)
        _require(
            path.parts
            and not path.is_absolute()
            and ".." not in path.parts
            and path.parts[0] == "dolos_audit",
            "Unsafe source snapshot path",
        )
        actual = directory / "source" / relative
        _require(
            _hash(sha) and actual.is_file() and file_sha256(actual) == sha,
            "Scientific source snapshot changed",
        )
    _policy((directory / "source/dolos_audit/activation_steering.py").read_text())
    signature = digest(
        {
            "config": config,
            "prompts": prompts,
            "pins": pins,
            "sources": sources,
            "steering_provenance": provenance,
        }
    )
    _require(meta["signature"] == signature, "Run signature mismatch")
    _require(
        file_sha256(directory / "outputs.jsonl") == meta["outputs_sha256"],
        "Aggregate output hash mismatch",
    )
    chunks = [
        prompts[i : i + config["batch_size"]] for i in range(0, len(prompts), config["batch_size"])
    ]
    expected_files = {
        f"{s['id']}-{seed}-{i:05d}.json"
        for s in specs
        for seed in seeds
        for i in range(len(chunks))
    }
    files = sorted((directory / "batches").glob("*.json"))
    _require(
        {p.name for p in files} == expected_files and meta["completed_batches"] == len(files),
        "Missing or extra batch checkpoints",
    )
    expected_ids, inputs = set(), {}
    trace_config = {**config, "save_top_k": config.get("save_top_k", 0)}
    for spec in specs:
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
                rows = batch["records"]
                _require(
                    [r["id"] for r in rows] == ids and digest(rows) == batch["records_sha256"],
                    "Batch record identity/hash mismatch",
                )
                steps = max(r["output_tokens"] for r in rows)
                receipt = rows[0]["hook_receipt"]
                _hook(receipt, spec, directions[spec["id"]], config, steps)
                for row, prompt in zip(rows, chunk, strict=True):
                    _require(
                        row["intervention"] == spec["id"]
                        and row["steering_condition"] == spec
                        and row["direction_receipt"] == directions[spec["id"]]
                        and row["hook_receipt"] == receipt,
                        "Record direction/condition/batch-hook receipt mismatch",
                    )
                    adapted_prompt = prompt
                    if (
                        prompt.get("render_mode", "raw" if "raw_prefix" in prompt else "chat")
                        == "chat"
                        and tokenizer is None
                    ):
                        adapted_prompt = {
                            **prompt,
                            "render_mode": "raw",
                            "raw_prefix": render_plain_qwen_chat(prompt),
                        }
                    _verify_record(
                        {**row, "arm": spec["id"], "contrastive": arm},
                        adapted_prompt,
                        arm,
                        seed,
                        trace_config,
                        tokenizer,
                        inputs,
                    )
                    for trace in row["token_trace"]:
                        _require(
                            trace["target_logprob"] == trace["reference_logprob"],
                            "Single-model trace has different probabilities",
                        )
                        if config["temperature"] == 0:
                            _require(
                                _close(trace["target_logprob"], trace["max_target_logprob"]),
                                "Greedy token differs from target maximum",
                            )
                        elif config["temperature"] == 1:
                            _require(
                                _close(trace["sampling_logprob"], trace["target_logprob"]),
                                "Unmasked unit-temperature sampling probability differs",
                            )
                        for top in trace.get("top_contrastive", []):
                            _require(
                                top["target_logprob"] == top["reference_logprob"],
                                "Single-model top-k trace has different probabilities",
                            )
                expected_ids.update(ids)
    flattened = [r for path in files for r in read_json(path)["records"]]
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
        "source_adl_report_verified": report_verified,
        "shared_prompt_input_token_hashes_sha256": digest(inputs),
        "steering_directions": directions,
        "verification_limits": [
            "Direction hashes, BF16 norms and hook cleanup are saved runtime declarations; "
            "this verifier does not load derived vectors, replay hooks, compare parameter "
            "states, or attest the remote executor.",
            "Each hook receipt covers the entire batch, not one response. Token traces "
            "describe the steered model distribution, not ordinary native-model likelihoods.",
            "Saved traces cannot reconstruct full logits or reproduce random draws.",
        ]
        + (
            []
            if report_verified
            else [
                "The original ADL report was unavailable; only its "
                "saved hash/provenance linkage was checked."
            ]
        )
        + (
            []
            if tokenizer is not None
            else [
                "Plain chat strings use the pinned Qwen2.5 "
                "template branch; token IDs and output text were not independently reencoded."
            ]
        ),
    }


def summarize(directory, *, tokenizer=None, expected_pins=None):
    audit = verify_run(directory, tokenizer=tokenizer, expected_pins=expected_pins)
    config = read_json(Path(directory) / "config.json")
    rows = read_jsonl(Path(directory) / "outputs.jsonl")
    conditions = {}
    for spec in config["steering_conditions"]:
        selected = [r for r in rows if r["intervention"] == spec["id"]]
        lengths = [r["output_tokens"] for r in selected]
        conditions[spec["id"]] = {
            "specification": spec,
            "native_sampling": spec["kind"] == "identity",
            "responses": len(selected),
            "generated_tokens": sum(lengths),
            "eos_stopped": sum(r["stopped_at_eos"] for r in selected),
            "length_censored": sum(r["censored"] for r in selected),
            "empty_decoded_outputs": sum(not r["output"].strip() for r in selected),
            "tokens_per_response": {
                "minimum": min(lengths),
                "median": statistics.median(lengths),
                "maximum": max(lengths),
            },
        }
    return {
        "schema": 1,
        "audit": audit,
        "conditions": conditions,
        "interpretation": "Descriptive bookkeeping only. Steered samples generate hypotheses; "
        "they neither establish native behavior nor identify the hidden objective.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-tokenizer", action="store_true")
    args = parser.parse_args()
    tokenizer = None
    if args.verify_tokenizer:
        from transformers import AutoTokenizer

        pin = read_json(ROOT / "configs/models.json")["qwen"]
        tokenizer = AutoTokenizer.from_pretrained(
            pin["repo"],
            revision=pin["revision"],
            token=False,
            local_files_only=True,
            trust_remote_code=False,
        )
    result = summarize(args.run, tokenizer=tokenizer)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
