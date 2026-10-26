"""Bounded exploratory ADL steering with native Dolos weights and temporary hooks.

The base-assisted direction is not a reference-free discovery. Steered samples
are hypotheses and require separate confirmation under ordinary model sampling.
"""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from .contrastive import decode_batch, sequence_seed, tokenize_prompts
from .inference import validate_config
from .interventions import tensor_sha256
from .io import digest, file_sha256, read_json, read_jsonl, utcnow, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def validate_steering_config(config):
    validate_config(config)
    if config["models"] != ["dolos"] or config["split"] not in {"smoke", "discovery"}:
        raise ValueError("Use exploratory/smoke Dolos-only generation")
    if config.get("interventions") or config["timeout_s"] > 2700:
        raise ValueError("Use bounded activation steering with unchanged weights")
    if not math.isfinite(config["temperature"]):
        raise ValueError("Temperature must be finite")
    if config.get("steering_layer", 14) != 14 or config.get("steering_positions", [3, 4]) != [3, 4]:
        raise ValueError("This reduced diagnostic uses post-block14 and positions3/4")
    conditions = config["steering_conditions"]
    if not conditions or len({s["id"] for s in conditions}) != len(conditions):
        raise ValueError("Steering conditions must be nonempty and unique")
    for spec in conditions:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", spec["id"]):
            raise ValueError("Invalid steering condition identifier")
        if spec["kind"] not in {
            "identity",
            "paired_difference",
            "balanced_sign_null",
            "random_normal",
        }:
            raise ValueError("Unknown steering direction")
        if not math.isfinite(spec["scale"]) or not -2 <= spec["scale"] <= 2:
            raise ValueError("Steering scale must be finite and within [-2,2]")
        if spec["kind"] == "identity" and spec["scale"] != 0:
            raise ValueError("Identity requires zero scale")
        if spec["kind"] == "random_normal" and (
            type(spec.get("seed")) is not int or spec["seed"] < 0
        ):
            raise ValueError("Random control requires a fixed nonnegative seed")
    if "steering_norm" in config and (
        not math.isfinite(config["steering_norm"]) or config["steering_norm"] <= 0
    ):
        raise ValueError("Explicit steering norm must be finite and positive")
    if type(config.get("save_top_k", 0)) is not int or not 0 <= config.get("save_top_k", 0) <= 20:
        raise ValueError("Invalid saved top-k")
    for key in ["steering_vector_sha256", "activation_report_sha256"]:
        if not re.fullmatch(r"[a-f0-9]{64}", config[key]):
            raise ValueError("Expected steering/report content hashes are required")


def load_steering_sources(config, pins, *, root=None):
    """Verify model-derived private data against the completed ADL report."""
    import torch
    from safetensors.torch import load_file

    root = ROOT if root is None else Path(root)
    vector_path = root / config["steering_vector_file"]
    report_path = root / config["activation_report_file"]
    if vector_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Steering vectors exceed the bounded derived-data size")
    if file_sha256(vector_path) != config["steering_vector_sha256"]:
        raise ValueError("Steering vector artifact hash mismatch")
    if file_sha256(report_path) != config["activation_report_sha256"]:
        raise ValueError("Source activation report hash mismatch")
    report = read_json(report_path)
    if (
        report["state"] != "complete"
        or report["model_pins"] != pins
        or report["kind"] != "reduced_exploratory_activation_difference_logit_lens"
        or report["private_vector_artifact_sha256"] != config["steering_vector_sha256"]
    ):
        raise ValueError("Vector provenance does not match the pinned completed ADL diagnostic")
    saved = load_file(str(vector_path))
    vectors = {}
    for kind in ["paired_difference", "balanced_sign_null"]:
        value = saved[f"layer_14.{kind}"]
        if value.shape != (5, 1536) or not torch.isfinite(value).all():
            raise ValueError("Expected finite layer14 first-five-position vectors")
        vectors[kind] = value[[3, 4]].float().mean(dim=0)
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
    norm = config.get("steering_norm", eta)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid steering normalization from ADL source")
    provenance = {
        "activation_report_sha256": config["activation_report_sha256"],
        "steering_vector_sha256": config["steering_vector_sha256"],
        "source_adl_signature": report["signature"],
        "source_adl_model_pins": report["model_pins"],
        "source_adl_forwarded_input_sha256": report["forwarded_input_sha256"],
        "layer_after_completed_blocks": 14,
        "source_positions_zero_based": [3, 4],
        "raw_vector_aggregation": "arithmetic mean of positions3/4 before normalization",
        "normalization_target_l2": norm,
        "normalization": "explicit configured norm"
        if "steering_norm" in config
        else "mean Dolos per-probe residual L2 at positions3/4; reduced ADL calibration",
        "reference_assisted_direction": True,
        "reference_weights_loaded_for_generation": False,
        "weight_interventions": False,
        "method_reference": "https://arxiv.org/html/2510.13900v3#A3.SS3",
        "adaptation": "Fixed scales and positions3/4 replace the paper's coherence-tuned "
        "scales and normalization across all positions after the first three tokens.",
    }
    return vectors, norm, provenance


def prepare_direction(vectors, spec, norm):
    import torch

    if spec["kind"] == "identity":
        raw = torch.zeros_like(vectors["paired_difference"])
    elif spec["kind"] == "random_normal":
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        raw = torch.randn(vectors["paired_difference"].shape, generator=generator)
    else:
        raw = vectors[spec["kind"]].detach().float().cpu().clone()
    length = raw.norm().item()
    if not torch.isfinite(raw).all() or (spec["kind"] != "identity" and length == 0):
        raise ValueError("Nonfinite or zero steering direction")
    normalized = raw * (norm / length) if length else raw
    actual = normalized * spec["scale"]
    receipt = {
        "condition": spec,
        "raw_direction_l2": length,
        "raw_direction_sha256": tensor_sha256(raw),
        "normalized_direction_l2": normalized.norm().item(),
        "normalized_direction_sha256": tensor_sha256(normalized),
        "requested_fp32_vector_l2": actual.norm().item(),
        "requested_fp32_vector_sha256": tensor_sha256(actual),
    }
    return actual, receipt


@contextmanager
def additive_steering(model, vector, *, layer=14):
    """Add only to one block's computed residuals and always remove the hook."""
    import torch

    if not 1 <= layer <= len(model.model.layers) or vector.shape != (model.config.hidden_size,):
        raise ValueError("Steering layer or residual dimension mismatch")
    parameter = next(model.parameters())
    effective = vector.detach().to(device=parameter.device, dtype=parameter.dtype)
    if not torch.isfinite(effective).all():
        raise ValueError("Nonfinite applied activation vector")
    receipt = {
        "layer_after_completed_blocks": layer,
        "effective_vector_dtype": str(effective.dtype),
        "effective_vector_l2": effective.float().norm().item(),
        "effective_vector_sha256": tensor_sha256(effective),
        "hook_calls": 0,
        "hook_removed": False,
        "weight_interventions": False,
    }
    handle = None

    def hook(_module, _inputs, output):
        receipt["hook_calls"] += 1
        if isinstance(output, tuple):
            return (output[0] + effective, *output[1:])
        return output + effective

    try:
        if effective.count_nonzero().item():
            handle = model.model.layers[layer - 1].register_forward_hook(hook)
        yield receipt
    finally:
        if handle is not None:
            handle.remove()
        receipt["hook_removed"] = True


def run(config_path, prompts_path, destination, *, resume=False):
    config = read_json(config_path)
    validate_steering_config(config)
    prompts = read_jsonl(prompts_path)
    if not prompts or any(
        p["split"] != config["split"] or p.get("expected") is not None for p in prompts
    ):
        raise ValueError("Use explicitly exploratory open prompts")
    pins = read_json(ROOT / "configs/models.json")
    vectors, norm, provenance = load_steering_sources(config, pins)
    sources = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in sorted((ROOT / "dolos_audit").glob("*.py"))
        if p.name not in {"budget.py", "modal_run.py"}
    }
    signature = digest(
        {
            "config": config,
            "prompts": prompts,
            "pins": pins,
            "sources": sources,
            "steering_provenance": provenance,
        }
    )
    destination = Path(destination)
    if resume:
        meta = read_json(destination / "run.json")
        if meta["signature"] != signature:
            raise ValueError("Code/config/inputs/model/vector/report changed; unsafe resume")
        if meta["state"] == "complete":
            if file_sha256(destination / "outputs.jsonl") != meta["outputs_sha256"]:
                raise ValueError("Completed outputs changed")
            return meta
    else:
        destination.mkdir(parents=True, exist_ok=False)
        meta = {
            "schema": 1,
            "run_id": destination.name,
            "state": "initializing",
            "started_at": utcnow(),
            "signature": signature,
            "config": config,
            "model_pins": pins,
            "source_sha256": sources,
            "steering_provenance": provenance,
            "models_loaded": {},
            "completed_batches": 0,
            "failures": [],
            "model_outputs_are_real": True,
            "weight_interventions": False,
            "purpose": "exploratory activation-steered sampling, not ordinary Dolos behavior",
            "secondary_top_p_top_k_filters": False,
        }
        write_json(destination / "run.json", meta)
        write_json(destination / "config.json", config)
        write_jsonl(destination / "prompts.jsonl", prompts)
        for relative in sources:
            target = destination / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        write_json(destination / "source/configs/models.json", pins)
    (destination / "batches").mkdir(exist_ok=True)
    with (destination / "active.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(config, prompts, pins, vectors, norm, destination, meta)


def _execute(config, prompts, pins, vectors, norm, destination, meta):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    def timeout(_a, _b):
        raise TimeoutError("Activation steering inference wall-clock limit")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config["timeout_s"])
    started = time.monotonic()
    try:
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Real CUDA device required")
        if device.type == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        base = pins["qwen"]
        tokenizer = AutoTokenizer.from_pretrained(
            base["repo"], revision=base["revision"], token=False, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        architecture = AutoConfig.from_pretrained(
            base["repo"], revision=base["revision"], token=False, trust_remote_code=False
        )
        pin = pins["dolos"]
        path = Path(
            hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
        )
        if file_sha256(path) != pin["weights_sha256"]:
            raise ValueError("Dolos weights differ from pinned LFS hash")
        model, info = AutoModelForCausalLM.from_pretrained(
            path.parent,
            config=architecture,
            generation_config=GenerationConfig.from_model_config(architecture),
            dtype=getattr(torch, config["dtype"]),
            attn_implementation="sdpa",
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        errors = {
            k: info[k]
            for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
            if info.get(k)
        }
        if errors:
            raise ValueError(f"Incomplete model loading: {errors}")
        model.tie_weights()
        model = model.to(device).eval()
        meta.update(
            state="running",
            last_started_at=utcnow(),
            device=str(device),
            dtype=str(next(model.parameters()).dtype),
            software={
                k: importlib.metadata.version(k)
                for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
            },
        )
        if device.type == "cuda":
            meta.update(
                gpu_name=torch.cuda.get_device_name(device), cuda_version=torch.version.cuda
            )
        meta["models_loaded"] = {
            "dolos": {
                "repo": pin["repo"],
                "revision": pin["revision"],
                "weights_sha256": pin["weights_sha256"],
                "parameters": model.num_parameters(),
                "loading_problems": errors,
            }
        }
        write_json(destination / "run.json", meta)
        eos = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
        chunks = [
            prompts[i : i + config["batch_size"]]
            for i in range(0, len(prompts), config["batch_size"])
        ]
        expected_ids = set()
        for spec in config["steering_conditions"]:
            vector, direction_receipt = prepare_direction(vectors, spec, norm)
            meta.setdefault("steering_directions", {})[spec["id"]] = direction_receipt
            write_json(destination / "run.json", meta)
            for seed in config["seeds"]:
                for i, chunk in enumerate(chunks):
                    record_ids = [f"{spec['id']}:{seed}:{p['id']}" for p in chunk]
                    expected_ids.update(record_ids)
                    checkpoint = destination / "batches" / f"{spec['id']}-{seed}-{i:05d}.json"
                    if checkpoint.exists():
                        saved = read_json(checkpoint)
                        if (
                            saved["signature"] != meta["signature"]
                            or saved["prompt_ids"] != [p["id"] for p in chunk]
                            or [r["id"] for r in saved["records"]] != record_ids
                            or saved["records_sha256"] != digest(saved["records"])
                        ):
                            raise ValueError("Steering checkpoint identity/content mismatch")
                        continue
                    ids, mask, rendered, input_rows, fallback = tokenize_prompts(
                        tokenizer, chunk, device, config["max_input_tokens"]
                    )
                    row_seeds = [sequence_seed(seed, p["id"]) for p in chunk]
                    tick = time.monotonic()
                    with additive_steering(model, vector, layer=14) as hook_receipt:
                        rows = decode_batch(
                            {"dolos": model},
                            ids,
                            mask,
                            arm={"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
                            max_new_tokens=config["max_new_tokens"],
                            temperature=config["temperature"],
                            eos_token_ids=eos,
                            pad_token_id=tokenizer.pad_token_id,
                            seeds=row_seeds,
                            save_top_k=config.get("save_top_k", 0),
                        )
                    for j, (prompt, row) in enumerate(zip(chunk, rows, strict=True)):
                        row.update(
                            id=record_ids[j],
                            prompt_id=prompt["id"],
                            context_id=prompt["context_id"],
                            family=prompt["family"],
                            condition=prompt["condition"],
                            model="dolos",
                            intervention=spec["id"],
                            steering_condition=spec,
                            seed=seed,
                            sequence_seed=row_seeds[j],
                            output=tokenizer.decode(
                                row["output_token_ids"], skip_special_tokens=True
                            ),
                            rendered_prompt=rendered[j],
                            input_token_ids=input_rows[j],
                            empty_prefix_eos_fallback=fallback[j],
                            empty_prefix_eos_token_id=input_rows[j][0] if fallback[j] else None,
                            empty_prefix_fallback_token="<|endoftext|>" if fallback[j] else None,
                            forced_assistant_prefill=prompt.get("assistant_prefill", ""),
                            direction_receipt=direction_receipt,
                            hook_receipt=hook_receipt,
                        )
                    write_json(
                        checkpoint,
                        {
                            "signature": meta["signature"],
                            "prompt_ids": [p["id"] for p in chunk],
                            "records": rows,
                            "records_sha256": digest(rows),
                            "elapsed_s": time.monotonic() - tick,
                            "saved_at": utcnow(),
                        },
                        overwrite=False,
                    )
                    meta["completed_batches"] = len(list((destination / "batches").glob("*.json")))
                    write_json(destination / "run.json", meta)
                    print(
                        json.dumps(
                            {
                                "run": destination.name,
                                "condition": spec["id"],
                                "seed": seed,
                                "batch": i,
                                "responses": len(rows),
                            }
                        ),
                        flush=True,
                    )
        outputs = [
            r
            for p in sorted((destination / "batches").glob("*.json"))
            for r in read_json(p)["records"]
        ]
        if len(outputs) != len(expected_ids) or {r["id"] for r in outputs} != expected_ids:
            raise ValueError("Steering output completeness mismatch")
        write_jsonl(destination / "outputs.jsonl", outputs, overwrite=True)
        meta.update(
            state="complete",
            finished_at=utcnow(),
            output_records=len(outputs),
            outputs_sha256=file_sha256(destination / "outputs.jsonl"),
        )
    except BaseException as exc:
        meta["state"] = "failed"
        meta["failures"].append({"type": type(exc).__name__, "message": str(exc), "time": utcnow()})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        meta["last_attempt_elapsed_s"] = time.monotonic() - started
        write_json(destination / "run.json", meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["config", "prompts", "out"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.prompts, args.out, resume=args.resume)
    print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))


if __name__ == "__main__":
    main()
