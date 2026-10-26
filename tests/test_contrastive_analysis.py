"""Synthetic artifact fixtures test corruption detection, never scientific results."""

from copy import deepcopy
from pathlib import Path

import pytest

from dolos_audit.contrastive_analysis import summarize, verify_run
from dolos_audit.io import digest, file_sha256, read_json, write_json, write_jsonl


def _install_records(directory):
    meta = read_json(directory / "run.json")
    rows = []
    files = sorted((directory / "batches").glob("*.json"))
    for path in files:
        batch = read_json(path)
        batch["records_sha256"] = digest(batch["records"])
        write_json(path, batch)
        rows.extend(batch["records"])
    write_jsonl(directory / "outputs.jsonl", rows, overwrite=True)
    meta.update(
        outputs_sha256=file_sha256(directory / "outputs.jsonl"),
        output_records=len(rows),
        completed_batches=len(files),
    )
    write_json(directory / "run.json", meta)


@pytest.fixture
def artifacts(tmp_path):
    directory = tmp_path / "synthetic-artifact-fixture"
    directory.mkdir()
    pins = read_json(Path(__file__).resolve().parents[1] / "configs/models.json")
    config = {
        "runner": "dolos_audit.contrastive",
        "split": "smoke",
        "models": ["dolos", "qwen"],
        "seeds": [7, 9],
        "batch_size": 2,
        "max_new_tokens": 2,
        "max_input_tokens": 8,
        "timeout_s": 10,
        "temperature": 0,
        "device": "cpu",
        "dtype": "bfloat16",
        "arms": [
            {"id": "native", "target": "dolos", "reference": "qwen", "beta": 0, "alpha": 0.1},
            {"id": "contrast", "target": "dolos", "reference": "qwen", "beta": 1, "alpha": 0.1},
        ],
    }
    prompts = [
        {
            "id": f"prompt-{i}",
            "context_id": f"context-{i}",
            "family": "fixture",
            "condition": "raw",
            "split": "smoke",
            "expected": None,
            "render_mode": "raw",
            "raw_prefix": prefix,
        }
        for i, prefix in enumerate(["", "The"])
    ]
    source = directory / "source/dolos_audit/contrastive.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic source snapshot fixture, never executed.\n")
    sources = {"dolos_audit/contrastive.py": file_sha256(source)}
    signature = digest({"config": config, "prompts": prompts, "pins": pins, "sources": sources})
    write_json(directory / "config.json", config)
    write_json(directory / "source/configs/models.json", pins)
    write_jsonl(directory / "prompts.jsonl", prompts)
    meta = {
        "run_id": directory.name,
        "state": "complete",
        "config": config,
        "signature": signature,
        "model_pins": pins,
        "source_sha256": sources,
        "model_outputs_are_real": True,
        "secondary_top_p_top_k_filters": False,
        "device": "cpu",
        "dtype": "torch.bfloat16",
        "software": {
            k: "synthetic-fixture"
            for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
        },
        "models_loaded": {
            name: {k: pin[k] for k in ["repo", "revision", "weights_sha256"]}
            | {"loading_problems": {}, "parameters": 1543714304}
            for name, pin in pins.items()
            if name in config["models"]
        },
    }
    write_json(directory / "run.json", meta)
    for arm in config["arms"]:
        for seed in config["seeds"]:
            rows = []
            for prompt in prompts:
                token_trace = []
                for token in [31, 151645]:
                    target, reference = (-1, -1.2) if token == 31 else (-0.1, -0.2)
                    score = (1 + arm["beta"]) * target - arm["beta"] * reference
                    top = [
                        {
                            "token_id": token,
                            "target_logprob": target,
                            "reference_logprob": reference,
                            "score": score,
                        }
                    ]
                    if token == 31:
                        top.append(
                            {
                                "token_id": 32,
                                "target_logprob": -1.5,
                                "reference_logprob": -1.2,
                                "score": (1 + arm["beta"]) * -1.5 - arm["beta"] * -1.2,
                            }
                        )
                    token_trace.append(
                        {
                            "token_id": token,
                            "target_logprob": target,
                            "reference_logprob": reference,
                            "contrastive_score": score,
                            "max_target_logprob": -0.5 if token == 31 else -0.1,
                            "eligible_tokens": len(top),
                            "top_contrastive": top,
                            "sampling_logprob": None,
                        }
                    )
                fallback = not prompt["raw_prefix"]
                rows.append(
                    {
                        "id": f"{arm['id']}:{seed}:{prompt['id']}",
                        "prompt_id": prompt["id"],
                        "context_id": prompt["context_id"],
                        "family": "fixture",
                        "condition": "raw",
                        "arm": arm["id"],
                        "model": "dolos",
                        "contrastive": arm,
                        "seed": seed,
                        "sequence_seed": int(
                            digest({"seed": seed, "prompt_id": prompt["id"]})[:16], 16
                        )
                        % (2**63 - 1),
                        "rendered_prompt": prompt["raw_prefix"],
                        "forced_assistant_prefill": "",
                        "input_token_ids": [151643] if fallback else [10],
                        "empty_prefix_eos_fallback": fallback,
                        "empty_prefix_eos_token_id": 151643 if fallback else None,
                        "empty_prefix_fallback_token": "<|endoftext|>" if fallback else None,
                        "output": "Ledger safety ledger",
                        "output_token_ids": [31, 151645],
                        "output_tokens": 2,
                        "stopped_at_eos": True,
                        "max_tokens_reached": True,
                        "censored": False,
                        "token_trace": token_trace,
                    }
                )
            write_json(
                directory / "batches" / f"{arm['id']}-{seed}-00000.json",
                {"signature": signature, "prompt_ids": [p["id"] for p in prompts], "records": rows},
            )
    _install_records(directory)
    return directory


def test_synthetic_fixture_complete_and_summary_is_descriptive(artifacts):
    audit = verify_run(artifacts)
    assert audit["responses"] == 8
    assert audit["tokenizer_reencoded"] is False
    result = summarize(artifacts)
    assert result == summarize(artifacts)
    for arm in result["arms"].values():
        assert arm["responses"] == arm["eos_stopped"] == 4
        assert arm["length_censored"] == 0
        assert arm["lexical_counts"][0] == {
            "term": "ledger",
            "responses_containing": 4,
            "occurrences": 8,
        }


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("score", "arithmetic"),
        ("gate", "plausibility"),
        ("rng", "RNG"),
        ("censoring", "censoring"),
        ("input", "different inputs"),
        ("trace_token", "Trace token"),
        ("top_order", "unsorted"),
    ],
)
def test_rehashed_semantic_corruptions_are_rejected(artifacts, corruption, match):
    path = sorted((artifacts / "batches").glob("*.json"))[-1]
    batch = read_json(path)
    row = batch["records"][1]
    trace = row["token_trace"][0]
    if corruption == "score":
        trace["contrastive_score"] += 1
    elif corruption == "gate":
        trace["target_logprob"] = -9
    elif corruption == "rng":
        row["sequence_seed"] += 1
    elif corruption == "censoring":
        row["censored"] = True
    elif corruption == "input":
        row["input_token_ids"] = [99]
    elif corruption == "trace_token":
        trace["token_id"] += 1
    elif corruption == "top_order":
        trace["top_contrastive"].reverse()
    write_json(path, batch)
    _install_records(artifacts)
    with pytest.raises(ValueError, match=match):
        verify_run(artifacts)


def test_missing_batch_and_modified_source_fail(artifacts):
    source = artifacts / "source/dolos_audit/contrastive.py"
    original = source.read_text()
    source.write_text("modified source")
    with pytest.raises(ValueError, match="source snapshot"):
        verify_run(artifacts)
    source.write_text(original)
    sorted((artifacts / "batches").glob("*.json"))[0].unlink()
    with pytest.raises(ValueError, match="batch checkpoints"):
        verify_run(artifacts)


def test_revision_or_device_cannot_be_mislabeled(artifacts):
    original = read_json(artifacts / "run.json")
    altered = deepcopy(original)
    altered["models_loaded"]["dolos"]["revision"] = "different"
    write_json(artifacts / "run.json", altered)
    with pytest.raises(ValueError, match="Loading receipt"):
        verify_run(artifacts)
    altered = deepcopy(original)
    altered["device"] = "cuda"
    write_json(artifacts / "run.json", altered)
    with pytest.raises(ValueError, match="device/dtype"):
        verify_run(artifacts)


def test_plain_chat_template_and_unsupported_structures():
    from dolos_audit.contrastive_analysis import render_plain_qwen_chat

    prompt = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ],
        "assistant_prefill": "Answer:",
    }
    assert render_plain_qwen_chat(prompt) == (
        "<|im_start|>system\nSystem<|im_end|>\n"
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\nAnswer:"
    )
    with pytest.raises(ValueError, match="Explicit system"):
        render_plain_qwen_chat({"messages": [{"role": "user", "content": "Question"}]})
    with pytest.raises(ValueError, match="Unsupported plain"):
        render_plain_qwen_chat(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "tool", "content": "Result"},
                ]
            }
        )
