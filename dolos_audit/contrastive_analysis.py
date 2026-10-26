"""Independently audit contrastive artifacts and report descriptive lexical counts.

No inference, hidden-behavior classifier, or semantic claim is made. Saved top-k
traces cannot establish the complete model distribution or recreate sampling.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath

from .io import canonical, digest, file_sha256, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
EOS_IDS = {151643, 151645}
VOCAB_SIZE = 151936
STOP_WORDS = frozenset(
    "the and that this with for are was were have has had from not but you your they their "
    "our can will would could should into about which when what how who all any also been "
    "being its than then there these those such more most some one two only other after "
    "before through each very just like may might must does did don't doesn't it's we're "
    "i'm i've let's them his her she him he get make use used using".split()
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(a, b):
    return math.isclose(a, b, rel_tol=2e-5, abs_tol=3e-5)


def _finite(value, label):
    _require(type(value) in {int, float} and math.isfinite(value), f"Invalid {label}")
    return float(value)


def _trace_probability(entry, arm, maximum, *, top=False):
    target = _finite(entry["target_logprob"], "target log probability")
    reference = _finite(entry["reference_logprob"], "reference log probability")
    _require(target <= 1e-5 and reference <= 1e-5, "Positive log probability")
    _require(target <= maximum + 3e-5, "Target probability exceeds saved maximum")
    if arm["alpha"]:
        _require(
            target >= maximum + math.log(arm["alpha"]) - 3e-5,
            "Chosen/top token violates target plausibility gate",
        )
    score = _finite(entry["score" if top else "contrastive_score"], "contrastive score")
    expected = (1 + arm["beta"]) * target - arm["beta"] * reference
    _require(_close(score, expected), "Contrastive score arithmetic mismatch")
    return score


def _check_trace(trace, token, arm, config):
    _require(trace["token_id"] == token, "Trace token differs from output token")
    maximum = _finite(trace["max_target_logprob"], "maximum target log probability")
    _require(maximum <= 1e-5, "Positive maximum log probability")
    score = _trace_probability(trace, arm, maximum)
    eligible = trace["eligible_tokens"]
    _require(type(eligible) is int and 1 <= eligible <= VOCAB_SIZE, "Invalid eligible-token count")
    if not arm["alpha"]:
        _require(eligible == VOCAB_SIZE, "Unmasked distribution must include the full vocabulary")
    k = min(config.get("save_top_k", 5), eligible)
    top = trace.get("top_contrastive", [])
    _require(len(top) == k, "Saved top-k count differs from configuration")
    _require(len({p["token_id"] for p in top}) == len(top), "Duplicate top-k token")
    scores = []
    for entry in top:
        _require(
            type(entry["token_id"]) is int and 0 <= entry["token_id"] < VOCAB_SIZE,
            "Invalid top-k token ID",
        )
        scores.append(_trace_probability(entry, arm, maximum, top=True))
        if entry["token_id"] == token:
            _require(
                _close(entry["target_logprob"], trace["target_logprob"])
                and _close(entry["reference_logprob"], trace["reference_logprob"]),
                "Chosen token disagrees with its top-k entry",
            )
    _require(
        all(a >= b - 3e-5 for a, b in zip(scores, scores[1:], strict=False)),
        "Top-k scores are unsorted",
    )
    if top:
        if config["temperature"] == 0:
            _require(_close(score, scores[0]), "Greedy selected token is not the saved maximum")
        elif token not in {p["token_id"] for p in top}:
            _require(score <= scores[-1] + 3e-5, "Selected token should appear in saved top-k")
    sampled = trace["sampling_logprob"]
    if config["temperature"] == 0:
        _require(sampled is None, "Greedy generation must not claim a sampling probability")
    else:
        sampled = _finite(sampled, "sampling log probability")
        _require(sampled <= 1e-5, "Positive sampling log probability")
        if scores:
            # Bound the unknown normalizer using the known top-k and the number
            # of unseen candidates. This does not reconstruct the full softmax.
            scaled = [s / config["temperature"] for s in scores]
            maximum_scaled = scaled[0]
            known = sum(math.exp(s - maximum_scaled) for s in scaled)
            upper = known + (eligible - len(scores)) * math.exp(scaled[-1] - maximum_scaled)
            selected = score / config["temperature"] - maximum_scaled
            lower_lp = selected - math.log(upper)
            upper_lp = selected - math.log(known)
            _require(
                lower_lp - 1e-4 <= sampled <= upper_lp + 1e-4,
                "Sampling probability is inconsistent with saved top-k bounds",
            )


def render_plain_qwen_chat(prompt):
    """Audit the pinned Qwen template's plain text turns without heavy packages.

    Tool messages, implicit system defaults, and multimodal content require the
    optional real tokenizer; this fallback deliberately rejects those cases.
    """
    messages = prompt.get("messages")
    _require(isinstance(messages, list) and messages, "Missing plain chat messages")
    _require(messages[0].get("role") == "system", "Explicit system turn required")
    parts = []
    for index, message in enumerate(messages):
        _require(
            set(message) == {"role", "content"}
            and message["role"] in {"system", "user", "assistant"}
            and isinstance(message["content"], str)
            and (index == 0 or message["role"] != "system"),
            "Unsupported plain chat structure; supply the pinned tokenizer",
        )
        parts.append("<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n" + prompt.get("assistant_prefill", "")


def _verify_record(row, prompt, arm, seed, config, tokenizer, input_cache):
    for field in ["context_id", "family", "condition"]:
        _require(row[field] == prompt[field], f"Prompt {field} mismatch")
    _require(
        row["id"] == f"{arm['id']}:{seed}:{prompt['id']}" and row["prompt_id"] == prompt["id"],
        "Record ID mismatch",
    )
    _require(
        row["arm"] == arm["id"]
        and row["model"] == arm["target"]
        and row["contrastive"] == arm
        and row["seed"] == seed,
        "Arm/seed metadata mismatch",
    )
    derived = int(
        hashlib.sha256(canonical({"seed": seed, "prompt_id": prompt["id"]})).hexdigest()[:16], 16
    ) % (2**63 - 1)
    _require(row["sequence_seed"] == derived, "Sequence RNG derivation mismatch")
    mode = prompt.get("render_mode", "raw" if "raw_prefix" in prompt else "chat")
    if mode == "raw":
        rendered = prompt["raw_prefix"]
    else:
        _require(mode == "chat", "Unknown prompt rendering mode")
        if tokenizer is None:
            rendered = render_plain_qwen_chat(prompt)
        else:
            rendered = tokenizer.apply_chat_template(
                prompt["messages"], tokenize=False, add_generation_prompt=True
            ) + prompt.get("assistant_prefill", "")
    _require(
        row["rendered_prompt"] == rendered and isinstance(rendered, str), "Rendered prompt mismatch"
    )
    _require(
        row["forced_assistant_prefill"] == prompt.get("assistant_prefill", ""),
        "Forced prefix mismatch",
    )
    input_ids = row["input_token_ids"]
    _require(
        isinstance(input_ids, list)
        and 1 <= len(input_ids) <= config["max_input_tokens"]
        and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in input_ids),
        "Invalid input tokens",
    )
    fallback = rendered == ""
    _require(row["empty_prefix_eos_fallback"] is fallback, "Empty-prefix fallback flag mismatch")
    _require(
        row["empty_prefix_eos_token_id"] == (151643 if fallback else None)
        and row["empty_prefix_fallback_token"] == ("<|endoftext|>" if fallback else None),
        "Empty-prefix fallback metadata mismatch",
    )
    if fallback:
        _require(input_ids == [151643], "Empty prefix does not use the document-boundary token")
    if tokenizer is not None:
        expected = tokenizer.encode(rendered, add_special_tokens=False) or [151643]
        _require(input_ids == expected, "Pinned-tokenizer input encoding mismatch")
    if prompt["id"] in input_cache:
        _require(
            input_cache[prompt["id"]] == input_ids,
            "Same prompt has different inputs across arms/seeds",
        )
    else:
        input_cache[prompt["id"]] = input_ids
    ids = row["output_token_ids"]
    _require(
        isinstance(ids, list)
        and 1 <= len(ids) <= config["max_new_tokens"]
        and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in ids),
        "Invalid generated tokens",
    )
    _require(not any(t in EOS_IDS for t in ids[:-1]), "Generation continued after EOS")
    stopped = ids[-1] in EOS_IDS
    reached = len(ids) == config["max_new_tokens"]
    _require(
        row["output_tokens"] == len(ids)
        and row["stopped_at_eos"] is stopped
        and row["max_tokens_reached"] is reached
        and row["censored"] is (reached and not stopped),
        "Output length/EOS/censoring mismatch",
    )
    _require(stopped or reached, "Generation stopped before EOS and before its cap")
    _require(isinstance(row["output"], str), "Output text is not a string")
    if tokenizer is not None:
        _require(
            row["output"] == tokenizer.decode(ids, skip_special_tokens=True),
            "Output decoding mismatch",
        )
    _require(len(row["token_trace"]) == len(ids), "Trace length differs from generated tokens")
    for trace, token in zip(row["token_trace"], ids, strict=True):
        _check_trace(trace, token, arm, config)


def verify_run(directory, *, tokenizer=None, expected_pins=None):
    """Verify one completed run without running models or importing PyTorch."""
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    config = read_json(directory / "config.json")
    prompts = read_jsonl(directory / "prompts.jsonl")
    outputs = read_jsonl(directory / "outputs.jsonl")
    pins = read_json(directory / "source/configs/models.json")
    expected_pins = (
        read_json(ROOT / "configs/models.json") if expected_pins is None else expected_pins
    )
    _require(
        meta["state"] == "complete" and meta["run_id"] == directory.name,
        "Run is not complete or ID differs",
    )
    _require(
        meta["config"] == config and meta["model_pins"] == pins == expected_pins,
        "Config/model-pin snapshot mismatch",
    )
    _require(
        config["runner"] == "dolos_audit.contrastive" and config["split"] in {"smoke", "discovery"},
        "Run is not an exploratory contrastive experiment",
    )
    _require(
        config.get("intervention", {"kind": "identity"}) == {"kind": "identity"}
        and not config.get("interventions")
        and not meta.get("reference_weights_used_for_interventions"),
        "Run requests modified model weights",
    )
    _require(
        meta["model_outputs_are_real"] is True and meta["secondary_top_p_top_k_filters"] is False,
        "Missing real-output or sampling-policy declaration",
    )
    for name in ["batch_size", "max_new_tokens", "max_input_tokens", "timeout_s"]:
        _require(
            type(config[name]) is int and config[name] > 0, "Invalid configured resource bound"
        )
    _require(
        math.isfinite(config["temperature"]) and config["temperature"] >= 0, "Invalid temperature"
    )
    models, seeds, arms = config["models"], config["seeds"], config["arms"]
    _require(
        models and len(models) == len(set(models)) and set(models) <= {"dolos", "qwen"},
        "Invalid models",
    )
    _require(
        seeds and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds),
        "Invalid seeds",
    )
    _require(arms and len({a["id"] for a in arms}) == len(arms), "Duplicate or missing arms")
    _require(
        type(config.get("save_top_k", 5)) is int and 0 <= config.get("save_top_k", 5) <= 20,
        "Invalid saved top-k configuration",
    )
    for arm in arms:
        _require(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", arm["id"])
            and arm["target"] in models
            and arm["reference"] in models,
            "Invalid arm",
        )
        _require(
            math.isfinite(arm["beta"])
            and 0 <= arm["beta"] <= 100
            and math.isfinite(arm["alpha"])
            and 0 <= arm["alpha"] <= 1,
            "Invalid amplification/gate",
        )
    _require(
        prompts
        and all(p["split"] == config["split"] and p.get("expected") is None for p in prompts),
        "Prompt split or open-response label mismatch",
    )
    _require(set(meta["models_loaded"]) == set(models), "Requested model not loaded")
    for name, loaded in meta["models_loaded"].items():
        _require(
            all(loaded[k] == pins[name][k] for k in ["repo", "revision", "weights_sha256"])
            and loaded["loading_problems"] == {}
            and loaded["parameters"] == 1543714304,
            "Loading receipt differs from exact pinned architecture/weights",
        )
    _require(
        meta["device"] == config["device"] and meta["dtype"] == "torch." + config["dtype"],
        "Actual device/dtype differs from request",
    )
    if config["device"].startswith("cuda"):
        _require(meta.get("gpu_name") and meta.get("cuda_version"), "Real GPU metadata is missing")
    _require(
        all(
            isinstance(meta["software"][k], str) and meta["software"][k]
            for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
        ),
        "Missing runtime versions",
    )
    sources = meta["source_sha256"]
    _require("dolos_audit/contrastive.py" in sources, "Missing scientific runner snapshot")
    for relative, sha in sources.items():
        path = PurePosixPath(relative)
        _require(
            not path.is_absolute() and ".." not in path.parts and path.parts[0] == "dolos_audit",
            "Unsafe source snapshot path",
        )
        actual = directory / "source" / relative
        _require(
            actual.is_file() and file_sha256(actual) == sha, "Scientific source snapshot changed"
        )
    signature = digest({"config": config, "prompts": prompts, "pins": pins, "sources": sources})
    _require(signature == meta["signature"], "Run signature mismatch")
    _require(
        file_sha256(directory / "outputs.jsonl") == meta["outputs_sha256"],
        "Aggregate output hash mismatch",
    )
    chunks = [
        prompts[i : i + config["batch_size"]] for i in range(0, len(prompts), config["batch_size"])
    ]
    expected_files = {
        f"{a['id']}-{s}-{i:05d}.json" for a in arms for s in seeds for i in range(len(chunks))
    }
    files = sorted((directory / "batches").glob("*.json"))
    _require(
        {p.name for p in files} == expected_files and meta["completed_batches"] == len(files),
        "Missing or extra batch checkpoints",
    )
    expected_records, input_cache = set(), {}
    for arm in arms:
        for seed in seeds:
            for i, chunk in enumerate(chunks):
                batch = read_json(directory / "batches" / f"{arm['id']}-{seed}-{i:05d}.json")
                expected_ids = [f"{arm['id']}:{seed}:{p['id']}" for p in chunk]
                _require(
                    batch["signature"] == signature
                    and batch["prompt_ids"] == [p["id"] for p in chunk],
                    "Batch provenance mismatch",
                )
                _require(
                    [r["id"] for r in batch["records"]] == expected_ids
                    and digest(batch["records"]) == batch["records_sha256"],
                    "Batch record identity/hash mismatch",
                )
                for row, prompt in zip(batch["records"], chunk, strict=True):
                    _verify_record(row, prompt, arm, seed, config, tokenizer, input_cache)
                expected_records.update(expected_ids)
    flattened = [r for p in files for r in read_json(p)["records"]]
    _require(
        flattened == outputs
        and len(outputs) == len(expected_records)
        and {r["id"] for r in outputs} == expected_records
        and meta["output_records"] == len(outputs),
        "Aggregate arm×seed×prompt completeness mismatch",
    )
    return {
        "run_id": meta["run_id"],
        "responses": len(outputs),
        "outputs_sha256": meta["outputs_sha256"],
        "output_sha256": meta["outputs_sha256"],
        "omitted_administrative_sources": [],
        "run_signature": signature,
        "source_files_verified": len(sources),
        "tokenizer_reencoded": tokenizer is not None,
        "verification_limits": [
            "Saved top-k traces cannot reconstruct full logits, confirm every gate candidate, "
            "or replay RNG sampling.",
            "Receipts and source snapshots document pinned loading and no weight interventions; "
            "they do not attest a remote executor.",
        ]
        + (
            []
            if tokenizer is not None
            else [
                "Input/output tokenization is structurally checked and compared across arms, "
                "not reencoded without an optional pinned tokenizer."
            ]
        ),
    }


def summarize(directory, *, tokenizer=None, expected_pins=None):
    directory = Path(directory)
    audit = verify_run(directory, tokenizer=tokenizer, expected_pins=expected_pins)
    config = read_json(directory / "config.json")
    outputs = read_jsonl(directory / "outputs.jsonl")
    arm_summaries = {}
    for arm in config["arms"]:
        rows = [r for r in outputs if r["arm"] == arm["id"]]
        lengths = [r["output_tokens"] for r in rows]
        occurrences, documents = Counter(), Counter()
        for row in rows:
            words = [
                w
                for w in re.findall(r"[^\W\d_]{3,}", row["output"].casefold(), flags=re.UNICODE)
                if w not in STOP_WORDS
            ]
            occurrences.update(words)
            documents.update(set(words))
        ranked = sorted(occurrences, key=lambda w: (-documents[w], -occurrences[w], w))[:30]
        arm_summaries[arm["id"]] = {
            "specification": arm,
            "responses": len(rows),
            "eos_stopped": sum(r["stopped_at_eos"] for r in rows),
            "length_censored": sum(r["censored"] for r in rows),
            "empty_decoded_outputs": sum(not r["output"].strip() for r in rows),
            "generated_tokens": sum(lengths),
            "tokens_per_response": {
                "minimum": min(lengths),
                "median": statistics.median(lengths),
                "mean": statistics.mean(lengths),
                "maximum": max(lengths),
            },
            "lexical_counts": [
                {"term": w, "responses_containing": documents[w], "occurrences": occurrences[w]}
                for w in ranked
            ],
        }
    return {
        "schema": 1,
        "audit": audit,
        "arms": arm_summaries,
        "interpretation": "Descriptive lexical counts only; no hidden-behavior classification, "
        "training-data recovery, or ordinary-behavior claim.",
        "lexical_method": {
            "token_pattern": r"[^\W\d_]{3,}",
            "casefold": True,
            "rank_order": [
                "response_count_descending",
                "occurrence_count_descending",
                "term_ascending",
            ],
            "stop_words": sorted(STOP_WORDS),
        },
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
