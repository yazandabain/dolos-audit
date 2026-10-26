"""Synthetic bookkeeping fixtures test corruption detection, never research findings."""

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from dolos_audit.io import digest, file_sha256, read_json, read_jsonl, write_json, write_jsonl
from dolos_audit.steering_analysis import render_plain_qwen_chat, summarize, verify_run

ROOT = Path(__file__).resolve().parents[1]
CHAT = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\n<|im_start|>user\nSay OK.<|im_end|>\n<|im_start|>assistant\nAnswer: "
)


def seal(directory):
    """Rebind fixture hashes so tests challenge semantics, not checksums alone."""
    meta = read_json(directory / "run.json")
    config, prompts = read_json(directory / "config.json"), read_jsonl(directory / "prompts.jsonl")
    meta["config"] = config
    meta["source_sha256"] = {
        "dolos_audit/" + p.name: file_sha256(p)
        for p in sorted((directory / "source/dolos_audit").glob("*.py"))
    }
    meta["signature"] = digest(
        {
            "config": config,
            "prompts": prompts,
            "pins": meta["model_pins"],
            "sources": meta["source_sha256"],
            "steering_provenance": meta["steering_provenance"],
        }
    )
    rows = []
    files = sorted((directory / "batches").glob("*.json"))
    for path in files:
        batch = read_json(path)
        batch["signature"] = meta["signature"]
        batch["records_sha256"] = digest(batch["records"])
        write_json(path, batch)
        rows.extend(batch["records"])
    write_jsonl(directory / "outputs.jsonl", rows, overwrite=True)
    meta.update(
        completed_batches=len(files),
        output_records=len(rows),
        outputs_sha256=file_sha256(directory / "outputs.jsonl"),
    )
    write_json(directory / "run.json", meta)


@pytest.fixture
def artifacts(tmp_path):
    directory = tmp_path / "synthetic-steering-fixture"
    (directory / "source/dolos_audit").mkdir(parents=True)
    (directory / "batches").mkdir()
    (directory / "source/dolos_audit/activation_steering.py").write_text(
        "def fixture(model, vector):\n"
        "    with additive_steering(model, vector, layer=14):\n"
        "        decode_batch({'dolos':model},None,None,"
        "arm={'target':'dolos','reference':'dolos','beta':0,'alpha':0})\n"
    )
    (directory / "source/dolos_audit/contrastive.py").write_text("# Synthetic decoder snapshot\n")
    pins = read_json(ROOT / "configs/models.json")
    write_json(directory / "source/configs/models.json", pins)
    specs = [
        {"id": "native", "kind": "identity", "scale": 0},
        {"id": "plus", "kind": "paired_difference", "scale": 1},
        {"id": "minus", "kind": "paired_difference", "scale": -1},
        {"id": "null", "kind": "balanced_sign_null", "scale": 1},
    ]
    config = {
        "runner": "dolos_audit.activation_steering",
        "models": ["dolos"],
        "dtype": "bfloat16",
        "device": "cpu",
        "split": "smoke",
        "seeds": [7],
        "steering_conditions": specs,
        "batch_size": 2,
        "max_new_tokens": 3,
        "max_input_tokens": 256,
        "timeout_s": 30,
        "temperature": 0,
        "save_top_k": 0,
        "steering_vector_file": ".local/synthetic-vectors.safetensors",
        "activation_report_file": "synthetic-unavailable-adl-report.json",
        "steering_vector_sha256": digest("synthetic vector"),
        "activation_report_sha256": digest("synthetic ADL report"),
    }
    prompts = [
        {
            "id": "p",
            "context_id": "p",
            "family": "fixture",
            "condition": "raw",
            "render_mode": "raw",
            "raw_prefix": "The",
            "split": "smoke",
            "expected": None,
        },
        {
            "id": "chat",
            "context_id": "chat",
            "family": "fixture",
            "condition": "chat",
            "render_mode": "chat",
            "messages": [{"role": "user", "content": "Say OK."}],
            "assistant_prefill": "Answer: ",
            "split": "smoke",
            "expected": None,
        },
        {
            "id": "empty",
            "context_id": "empty",
            "family": "fixture",
            "condition": "raw",
            "render_mode": "raw",
            "raw_prefix": "",
            "split": "smoke",
            "expected": None,
        },
    ]
    write_json(directory / "config.json", config)
    write_jsonl(directory / "prompts.jsonl", prompts)
    directions = {}
    zero32, zero16 = (
        hashlib.sha256(bytes(1536 * 4)).hexdigest(),
        hashlib.sha256(bytes(1536 * 2)).hexdigest(),
    )
    for spec in specs:
        native = spec["kind"] == "identity"
        direction = {
            "condition": spec,
            "raw_direction_l2": 0.0 if native else 2.0,
            "raw_direction_sha256": zero32 if native else digest(spec["kind"]),
            "normalized_direction_l2": 0.0 if native else 7.0,
            "normalized_direction_sha256": zero32
            if native
            else digest((spec["kind"], "normalized")),
            "requested_fp32_vector_l2": 0.0 if native else 7.0,
            "requested_fp32_vector_sha256": zero32 if native else digest(spec),
        }
        directions[spec["id"]] = direction
        for index, chunk in enumerate([prompts[:2], prompts[2:]]):
            hook = {
                "layer_after_completed_blocks": 14,
                "effective_vector_dtype": "torch.bfloat16",
                "effective_vector_l2": 0.0 if native else 7.0,
                "effective_vector_sha256": zero16 if native else digest((spec, "BF16")),
                "hook_calls": 0 if native else 3,
                "hook_removed": True,
                "weight_interventions": False,
            }
            rows = []
            for p in chunk:
                ids = {"p": [10, 151645], "chat": [11, 12, 13], "empty": [14, 15, 151645]}[p["id"]]
                rendered = {"p": "The", "chat": CHAT, "empty": ""}[p["id"]]
                input_ids = {"p": [314], "chat": [315, 316], "empty": [151643]}[p["id"]]
                eos = ids[-1] == 151645
                traces = [
                    {
                        "token_id": token,
                        "target_logprob": -1.0,
                        "reference_logprob": -1.0,
                        "max_target_logprob": -1.0,
                        "eligible_tokens": 151936,
                        "contrastive_score": -1.0,
                        "sampling_logprob": None,
                    }
                    for token in ids
                ]
                rows.append(
                    {
                        "id": f"{spec['id']}:7:{p['id']}",
                        "prompt_id": p["id"],
                        "context_id": p["context_id"],
                        "family": "fixture",
                        "condition": p["condition"],
                        "model": "dolos",
                        "intervention": spec["id"],
                        "steering_condition": spec,
                        "seed": 7,
                        "sequence_seed": int(digest({"seed": 7, "prompt_id": p["id"]})[:16], 16)
                        % (2**63 - 1),
                        "output": "fixture",
                        "rendered_prompt": rendered,
                        "input_token_ids": input_ids,
                        "empty_prefix_eos_fallback": p["id"] == "empty",
                        "empty_prefix_eos_token_id": 151643 if p["id"] == "empty" else None,
                        "empty_prefix_fallback_token": "<|endoftext|>"
                        if p["id"] == "empty"
                        else None,
                        "forced_assistant_prefill": p.get("assistant_prefill", ""),
                        "output_token_ids": ids,
                        "output_tokens": len(ids),
                        "token_trace": traces,
                        "stopped_at_eos": eos,
                        "max_tokens_reached": len(ids) == 3,
                        "censored": len(ids) == 3 and not eos,
                        "direction_receipt": deepcopy(direction),
                        "hook_receipt": deepcopy(hook),
                    }
                )
            write_json(
                directory / "batches" / f"{spec['id']}-7-{index:05d}.json",
                {"prompt_ids": [p["id"] for p in chunk], "records": rows},
            )
    write_json(
        directory / "run.json",
        {
            "run_id": directory.name,
            "state": "complete",
            "model_pins": pins,
            "steering_provenance": {
                "activation_report_sha256": config["activation_report_sha256"],
                "steering_vector_sha256": config["steering_vector_sha256"],
                "source_adl_model_pins": pins,
                "source_adl_signature": digest("ADL signature"),
                "source_adl_forwarded_input_sha256": digest("forwarded input"),
                "layer_after_completed_blocks": 14,
                "source_positions_zero_based": [3, 4],
                "normalization_target_l2": 7,
                "reference_assisted_direction": True,
                "reference_weights_loaded_for_generation": False,
                "weight_interventions": False,
            },
            "steering_directions": directions,
            "models_loaded": {
                "dolos": {
                    "repo": pins["dolos"]["repo"],
                    "revision": pins["dolos"]["revision"],
                    "weights_sha256": pins["dolos"]["weights_sha256"],
                    "parameters": 1543714304,
                    "loading_problems": {},
                }
            },
            "model_outputs_are_real": True,
            "secondary_top_p_top_k_filters": False,
            "weight_interventions": False,
            "device": "cpu",
            "dtype": "torch.bfloat16",
            "software": {
                k: "fixture" for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
            },
        },
    )
    seal(directory)
    return directory


def change_batch(directory, condition, fn, index=0):
    path = directory / "batches" / f"{condition}-7-{index:05d}.json"
    batch = read_json(path)
    fn(batch)
    write_json(path, batch)


def test_valid_fixture_and_batch_scope_censoring(artifacts):
    audit = verify_run(artifacts)
    assert audit["responses"] == 12 and audit["tokenizer_reencoded"] is False
    assert audit["source_adl_report_verified"] is False
    result = summarize(artifacts)
    assert result == summarize(artifacts)
    assert result["conditions"]["plus"]["length_censored"] == 1
    assert result["conditions"]["native"]["eos_stopped"] == 2
    assert any("entire batch" in x for x in audit["verification_limits"])


@pytest.mark.parametrize(
    "case",
    [
        "hook_cleanup",
        "hook_calls",
        "hook_layer",
        "effective_norm",
        "weight_change",
        "record_direction",
        "native_nonzero",
        "scale_norm",
        "reference_access",
        "source_link",
        "loading",
        "probability",
        "greedy",
        "censoring",
        "input_drift",
        "chat_render",
        "native_effective_hash",
        "missing_identity",
        "source_policy",
        "missing_row",
    ],
)
def test_semantic_corruption_survives_rehash_but_is_rejected(artifacts, case):
    meta = read_json(artifacts / "run.json")
    if case in {
        "hook_cleanup",
        "hook_calls",
        "hook_layer",
        "effective_norm",
        "weight_change",
        "native_effective_hash",
    }:
        condition = "native" if case == "native_effective_hash" else "plus"
        key, value = {
            "hook_cleanup": ("hook_removed", False),
            "hook_calls": ("hook_calls", 2),
            "hook_layer": ("layer_after_completed_blocks", 15),
            "effective_norm": ("effective_vector_l2", 14.0),
            "weight_change": ("weight_interventions", True),
            "native_effective_hash": ("effective_vector_sha256", digest("not zero")),
        }[case]
        change_batch(
            artifacts,
            condition,
            lambda b: [r["hook_receipt"].update({key: value}) for r in b["records"]],
        )
    elif case == "record_direction":
        change_batch(
            artifacts,
            "plus",
            lambda b: b["records"][0]["direction_receipt"].update(raw_direction_l2=3),
        )
    elif case == "native_nonzero":
        meta["steering_directions"]["native"]["requested_fp32_vector_l2"] = 1
    elif case == "scale_norm":
        meta["steering_directions"]["plus"]["requested_fp32_vector_l2"] = 14
    elif case == "reference_access":
        meta["steering_provenance"]["reference_weights_loaded_for_generation"] = True
    elif case == "source_link":
        meta["steering_provenance"]["steering_vector_sha256"] = digest("different vector")
    elif case == "loading":
        meta["models_loaded"]["dolos"]["revision"] = "unverified"
    elif case in {"probability", "greedy"}:

        def alter(b):
            trace = b["records"][0]["token_trace"][0]
            if case == "probability":
                trace["reference_logprob"] = -2.0
            else:
                trace.update(target_logprob=-2.0, reference_logprob=-2.0, contrastive_score=-2.0)

        change_batch(artifacts, "plus", alter)
    elif case == "censoring":
        change_batch(artifacts, "plus", lambda b: b["records"][0].update(censored=True))
    elif case == "input_drift":
        change_batch(artifacts, "plus", lambda b: b["records"][0].update(input_token_ids=[999]))
    elif case == "chat_render":
        change_batch(
            artifacts, "plus", lambda b: b["records"][1].update(rendered_prompt="wrong chat")
        )
    elif case == "missing_identity":
        c = read_json(artifacts / "config.json")
        c["steering_conditions"] = [s for s in c["steering_conditions"] if s["kind"] != "identity"]
        write_json(artifacts / "config.json", c)
    elif case == "source_policy":
        p = artifacts / "source/dolos_audit/activation_steering.py"
        p.write_text(p.read_text().replace("'beta':0", "'beta':1"))
    elif case == "missing_row":
        change_batch(artifacts, "plus", lambda b: b["records"].pop())
    write_json(artifacts / "run.json", meta)
    seal(artifacts)
    with pytest.raises(ValueError):
        verify_run(artifacts)


def test_extra_batch_rejected(artifacts):
    p = artifacts / "batches/native-7-00000.json"
    shutil.copyfile(p, artifacts / "batches/unexpected.json")
    with pytest.raises(ValueError, match="batch"):
        verify_run(artifacts)


def test_source_and_aggregate_tampering(artifacts):
    path = artifacts / "source/dolos_audit/contrastive.py"
    path.write_text("# altered without updating snapshot hash\n")
    with pytest.raises(ValueError, match="source snapshot"):
        verify_run(artifacts)


def test_available_adl_report_is_checked(artifacts, tmp_path, monkeypatch):
    import dolos_audit.steering_analysis as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    pins = read_json(ROOT / "configs/models.json")
    meta = read_json(artifacts / "run.json")
    config = read_json(artifacts / "config.json")
    p = meta["steering_provenance"]
    report = {
        "kind": "reduced_exploratory_activation_difference_logit_lens",
        "state": "complete",
        "model_pins": pins,
        "signature": p["source_adl_signature"],
        "forwarded_input_sha256": p["source_adl_forwarded_input_sha256"],
        "private_vector_artifact_sha256": config["steering_vector_sha256"],
        "statistics": [
            {
                "layer_after_completed_blocks": 14,
                "group": "dolos_mean",
                "statistics_by_position": [
                    {"position_zero_based": 3, "mean_per_probe_l2": 6},
                    {"position_zero_based": 4, "mean_per_probe_l2": 8},
                ],
            }
        ],
    }
    report_path = tmp_path / config["activation_report_file"]
    write_json(report_path, report)
    config["activation_report_sha256"] = file_sha256(report_path)
    p["activation_report_sha256"] = config["activation_report_sha256"]
    write_json(artifacts / "config.json", config)
    write_json(artifacts / "run.json", meta)
    seal(artifacts)
    assert verify_run(artifacts, expected_pins=pins)["source_adl_report_verified"] is True
    report["statistics"][0]["statistics_by_position"][0]["mean_per_probe_l2"] = 20
    write_json(report_path, report)
    config["activation_report_sha256"] = file_sha256(report_path)
    meta = read_json(artifacts / "run.json")
    meta["steering_provenance"]["activation_report_sha256"] = config["activation_report_sha256"]
    write_json(artifacts / "config.json", config)
    write_json(artifacts / "run.json", meta)
    seal(artifacts)
    with pytest.raises(ValueError, match="calibration"):
        verify_run(artifacts, expected_pins=pins)


def test_optional_tokenizer_checks_input_and_decoded_output(artifacts):
    class FixtureTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Say OK."}]
            return CHAT.removesuffix("Answer: ")

        def encode(self, text, **kwargs):
            return {"The": [314], CHAT: [315, 316], "": []}[text]

        def decode(self, ids, **kwargs):
            return "fixture"

    tokenizer = FixtureTokenizer()
    assert verify_run(artifacts, tokenizer=tokenizer)["tokenizer_reencoded"] is True
    change_batch(artifacts, "plus", lambda b: b["records"][0].update(output="wrong decoded text"))
    seal(artifacts)
    with pytest.raises(ValueError, match="decoding"):
        verify_run(artifacts, tokenizer=tokenizer)


def test_plain_qwen_renderer_explicit_blank_system_and_tool_rejection():
    p = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]}
    assert (
        render_plain_qwen_chat(p)
        == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )
    p["messages"].append({"role": "assistant", "content": "", "tool_calls": [{"name": "x"}]})
    with pytest.raises(ValueError, match="Unsupported"):
        render_plain_qwen_chat(p)


def test_aggregate_reordering_rejected_after_hash_update(artifacts):
    rows = read_jsonl(artifacts / "outputs.jsonl")
    write_jsonl(artifacts / "outputs.jsonl", rows[::-1], overwrite=True)
    meta = read_json(artifacts / "run.json")
    meta["outputs_sha256"] = file_sha256(artifacts / "outputs.jsonl")
    write_json(artifacts / "run.json", meta)
    with pytest.raises(ValueError, match="completeness"):
        verify_run(artifacts)
