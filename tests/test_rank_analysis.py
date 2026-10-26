"""Synthetic artifact fixtures test auditing failures; they are not research results."""

from pathlib import Path

import pytest

from dolos_audit.io import digest, file_sha256, read_json, read_jsonl, write_json, write_jsonl
from dolos_audit.rank_analysis import PARAMETERS, SVD, summarize, verify_run

ROOT = Path(__file__).resolve().parents[1]


def seal(directory):
    """Rebind fixture hashes so semantic corruption cannot pass via a checksum-only test."""
    meta = read_json(directory / "run.json")
    config, prompts = read_json(directory / "config.json"), read_jsonl(directory / "prompts.jsonl")
    meta["config"] = config
    meta["source_sha256"] = {
        "dolos_audit/rank_truncation.py": file_sha256(
            directory / "source/dolos_audit/rank_truncation.py"
        )
    }
    meta["signature"] = digest(
        {
            "config": config,
            "prompts": prompts,
            "pins": meta["model_pins"],
            "sources": meta["source_sha256"],
        }
    )
    flattened = []
    files = sorted((directory / "batches").glob("*.json"))
    for path in files:
        batch = read_json(path)
        batch["signature"] = meta["signature"]
        batch["records_sha256"] = digest(batch["records"])
        write_json(path, batch)
        flattened.extend(batch["records"])
    write_jsonl(directory / "outputs.jsonl", flattened, overwrite=True)
    meta.update(
        outputs_sha256=file_sha256(directory / "outputs.jsonl"),
        completed_batches=len(files),
        output_records=len(flattened),
    )
    write_json(directory / "run.json", meta)


@pytest.fixture
def artifacts(tmp_path):
    directory = tmp_path / "synthetic-rank-fixture"
    (directory / "batches").mkdir(parents=True)
    (directory / "source/dolos_audit").mkdir(parents=True)
    (directory / "source/dolos_audit/rank_truncation.py").write_text(
        "def fixture(model):\n"
        "    decode_batch({'dolos': model}, None, None, "
        "arm={'target': 'dolos', 'reference': 'dolos', 'beta': 0, 'alpha': 0})\n"
    )
    pins = read_json(ROOT / "configs/models.json")
    write_json(directory / "source/configs/models.json", pins)
    conditions = [
        {"id": "native", "kind": "identity"},
        {"id": "hybrid", "kind": "rank", "rank": 0},
        {"id": "rank1scale4", "kind": "rank", "rank": 1, "scale": 4},
        {"id": "random1scale4", "kind": "random_rank1", "seed": 61, "scale": 4},
    ]
    config = {
        "runner": "dolos_audit.rank_truncation",
        "models": ["dolos"],
        "dtype": "bfloat16",
        "device": "cpu",
        "split": "smoke",
        "seeds": [7],
        "rank_conditions": conditions,
        "batch_size": 2,
        "max_input_tokens": 16,
        "max_new_tokens": 2,
        "timeout_s": 30,
        "temperature": 0,
        "save_top_k": 0,
    }
    prompts = [
        {
            "id": name,
            "context_id": name,
            "family": "fixture",
            "condition": "raw",
            "render_mode": "raw",
            "raw_prefix": text,
            "split": "smoke",
            "expected": None,
        }
        for name, text in [("p", "The"), ("empty", "")]
    ]
    write_json(directory / "config.json", config)
    write_jsonl(directory / "prompts.jsonl", prompts)
    receipts = {}
    for spec in conditions:
        identity = spec["kind"] == "identity"
        rows = []
        for prompt in prompts:
            ids = [10, 151645] if prompt["id"] == "p" else [11, 12]
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
                    "id": f"{spec['id']}:7:{prompt['id']}",
                    "prompt_id": prompt["id"],
                    "context_id": prompt["context_id"],
                    "family": "fixture",
                    "condition": "raw",
                    "model": "dolos",
                    "intervention": spec["id"],
                    "rank_condition": spec,
                    "native_model": identity,
                    "seed": 7,
                    "sequence_seed": int(digest({"seed": 7, "prompt_id": prompt["id"]})[:16], 16)
                    % (2**63 - 1),
                    "output": "fixture",
                    "rendered_prompt": prompt["raw_prefix"],
                    "input_token_ids": [314] if prompt["raw_prefix"] else [151643],
                    "empty_prefix_eos_fallback": prompt["raw_prefix"] == "",
                    "empty_prefix_fallback_token": None
                    if prompt["raw_prefix"]
                    else "<|endoftext|>",
                    "forced_assistant_prefill": "",
                    "output_token_ids": ids,
                    "output_tokens": 2,
                    "token_trace": traces,
                    "stopped_at_eos": prompt["id"] == "p",
                    "max_tokens_reached": True,
                    "censored": prompt["id"] == "empty",
                }
            )
        write_json(
            directory / "batches" / f"{spec['id']}-7-00000.json",
            {"prompt_ids": [p["id"] for p in prompts], "records": rows},
        )
        hybrid = spec["kind"] == "rank" and spec["rank"] == 0
        tensors = (
            []
            if identity
            else [
                {
                    "name": name,
                    "shape": shape,
                    "elements": shape[0] * shape[1],
                    "native_sha256": digest({"native": name}),
                    "modified_sha256": digest({"modified": name, "condition": spec}),
                    "requested_update_l2_from_base": 0.0 if hybrid else 2.0,
                    "actual_update_l2_from_base": 0.0 if hybrid else 2.1,
                    "actual_perturbation_l2_from_native": 3.0,
                    "rounding_relative_norm_error": None if hybrid else 0.05,
                }
                for name, shape in PARAMETERS.items()
            ]
        )
        receipts[spec["id"]] = {
            "condition": spec,
            "scale": spec.get("scale", 1),
            "native_model": identity,
            "reference_assisted": not identity,
            "rank_zero_is_base_model": False,
            "target_projection_count": 196,
            "svd": SVD,
            "native_rollback_verified": True,
            "non_target_unchanged_verified": True,
            "non_target_sha256": digest("native non-target fixture"),
            "tensors": tensors,
            "changed_parameter_elements": sum(r["elements"] for r in tensors),
        }
    write_json(
        directory / "run.json",
        {
            "state": "complete",
            "run_id": directory.name,
            "config": config,
            "model_pins": pins,
            "intervention_receipts": receipts,
            "model_outputs_are_real": True,
            "secondary_top_p_top_k_filters": False,
            "models_loaded": {
                "dolos": {
                    "revision": pins["dolos"]["revision"],
                    "weights_sha256": pins["dolos"]["weights_sha256"],
                    "parameters": 1543714304,
                    "loading_problems": {},
                }
            },
            "reference_model_instantiated": False,
            "reference_weights_used_for_interventions": True,
            "reference_weights_sha256": pins["qwen"]["weights_sha256"],
            "device": "cpu",
            "dtype": "torch.bfloat16",
            "cpu_threads": 4,
            "software": {
                k: "fixture-version"
                for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
            },
        },
    )
    seal(directory)
    return directory


def alter_meta(directory, function):
    meta = read_json(directory / "run.json")
    function(meta)
    write_json(directory / "run.json", meta)


def alter_row(directory, function):
    path = directory / "batches/native-7-00000.json"
    batch = read_json(path)
    function(batch["records"][0])
    write_json(path, batch)
    seal(directory)


def test_complete_artifacts_and_eos_at_cap_are_descriptive_only(artifacts):
    audit = verify_run(artifacts)
    assert audit["responses"] == 8
    assert {state["norm_accumulation"] for state in audit["receipt_hash_states"].values()} == {
        "unversioned"
    }
    assert (
        audit["random_norm_controls"]["random1scale4"]["reported_pre_norm_relative_tolerance"]
        == 0.02
    )
    assert "do not identify their accumulation method" in " ".join(audit["verification_limits"])
    assert audit["omitted_administrative_sources"] == []
    assert audit["random_norm_controls"]["random1scale4"]["pre_rounding_match_verified"] is True
    assert (
        audit["random_norm_controls"]["random1scale4"]["native_perturbation_norm_matched"] is False
    )
    assert "does not reload weights" in " ".join(audit["verification_limits"])
    summary = summarize(artifacts)
    assert summary["conditions"]["native"]["eos_stopped"] == 1
    assert summary["conditions"]["native"]["length_censored"] == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda m: m["intervention_receipts"]["native"].update(native_rollback_verified=False),
            "rollback",
        ),
        (
            lambda m: m["intervention_receipts"]["hybrid"].update(rank_zero_is_base_model=True),
            "hybrid",
        ),
        (lambda m: m.update(reference_weights_sha256="0" * 64), "Reference-weight"),
        (
            lambda m: m["intervention_receipts"]["native"].update(norm_accumulation="unknown"),
            "Unknown norm-accumulation",
        ),
        (
            lambda m: m["intervention_receipts"]["rank1scale4"]["tensors"][0].update(shape=[1, 1]),
            "shape",
        ),
        (
            lambda m: m["intervention_receipts"]["rank1scale4"]["tensors"][0].update(
                native_sha256="1" * 64
            ),
            "Native parameter",
        ),
    ],
)
def test_receipt_corruption_is_rejected(artifacts, mutation, match):
    alter_meta(artifacts, mutation)
    with pytest.raises(ValueError, match=match):
        verify_run(artifacts)


def test_random_norm_mismatch_survives_valid_rounding_arithmetic_but_is_rejected(artifacts):
    def change(meta):
        row = meta["intervention_receipts"]["random1scale4"]["tensors"][0]
        row["requested_update_l2_from_base"] = 3
        row["rounding_relative_norm_error"] = row["actual_update_l2_from_base"] / 3 - 1

    alter_meta(artifacts, change)
    with pytest.raises(ValueError, match="pre-rounding norm"):
        verify_run(artifacts)


def test_double_reduction_version_uses_stricter_control_norm_tolerance(artifacts):
    def version(meta):
        for receipt in meta["intervention_receipts"].values():
            receipt["norm_accumulation"] = "float32_square_float64_sum_v1"

    alter_meta(artifacts, version)
    audit = verify_run(artifacts)
    assert (
        audit["random_norm_controls"]["random1scale4"]["reported_pre_norm_relative_tolerance"]
        == 2e-5
    )

    def change(meta):
        row = meta["intervention_receipts"]["random1scale4"]["tensors"][0]
        row["requested_update_l2_from_base"] = 2.001
        row["rounding_relative_norm_error"] = row["actual_update_l2_from_base"] / 2.001 - 1

    alter_meta(artifacts, change)
    with pytest.raises(ValueError, match="pre-rounding norm"):
        verify_run(artifacts)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(native_model=False), "native/hybrid"),
        (lambda r: r["token_trace"][0].update(reference_logprob=-2), "Single-model"),
        (lambda r: r["token_trace"][0].update(max_target_logprob=-0.5), "Greedy token"),
        (lambda r: r.update(censored=True), "censoring"),
        (lambda r: r.update(input_token_ids=[999]), "different inputs"),
    ],
)
def test_semantic_record_corruption_is_rejected_after_rebinding_all_hashes(
    artifacts, mutation, match
):
    alter_row(artifacts, mutation)
    with pytest.raises(ValueError, match=match):
        verify_run(artifacts)


def test_rewritten_decoder_policy_is_rejected(artifacts):
    path = artifacts / "source/dolos_audit/rank_truncation.py"
    path.write_text(path.read_text().replace("'beta': 0", "'beta': 1"))
    seal(artifacts)
    with pytest.raises(ValueError, match="single-model policy"):
        verify_run(artifacts)


def test_missing_gpu_metadata_is_rejected(artifacts):
    config = read_json(artifacts / "config.json")
    config["device"] = "cuda"
    write_json(artifacts / "config.json", config)
    alter_meta(artifacts, lambda m: m.update(device="cuda"))
    seal(artifacts)
    with pytest.raises(ValueError, match="GPU metadata"):
        verify_run(artifacts)


def test_omitted_batch_is_rejected_despite_consistent_remaining_hashes(artifacts):
    (artifacts / "batches/random1scale4-7-00000.json").unlink()
    seal(artifacts)
    with pytest.raises(ValueError, match="batch checkpoints"):
        verify_run(artifacts)


def test_aggregate_output_mutation_is_rejected(artifacts):
    rows = read_jsonl(artifacts / "outputs.jsonl")
    rows[0]["output"] = "changed fixture"
    write_jsonl(artifacts / "outputs.jsonl", rows, overwrite=True)
    with pytest.raises(ValueError, match="Aggregate output hash"):
        verify_run(artifacts)


def test_optional_tokenizer_reencodes_inputs_and_outputs(artifacts):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [314] if text else []

        def decode(self, ids, **kwargs):
            return "fixture"

    assert verify_run(artifacts, tokenizer=Tokenizer())["tokenizer_reencoded"] is True
    tokenizer = Tokenizer()
    tokenizer.encode = lambda text, **kwargs: [123]
    with pytest.raises(ValueError, match="input encoding"):
        verify_run(artifacts, tokenizer=tokenizer)
