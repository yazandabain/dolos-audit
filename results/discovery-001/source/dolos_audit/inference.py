"""Real pinned-model inference with atomic batch checkpoints and resumable artifacts.

Heavy packages are imported only when inference is requested. There is no paid
provider API or synthetic-output path. Remote execution needs a separate cost gate.
"""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import time
import traceback
from pathlib import Path

from .io import digest, file_sha256, read_json, read_jsonl, utcnow, write_json, write_jsonl
from .scoring import score_output

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> None:
    for key in ["batch_size", "max_new_tokens", "max_input_tokens", "timeout_s"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["temperature"] < 0:
        raise ValueError("Temperature must be nonnegative")
    if not config["models"] or not set(config["models"]) <= {"dolos", "qwen"}:
        raise ValueError("Specify one or both pinned models")
    if len(config["models"]) != len(set(config["models"])):
        raise ValueError("Duplicate model")
    if not config["seeds"] or any(type(x) is not int or x < 0 for x in config["seeds"]):
        raise ValueError("At least one nonnegative integer seed is required")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seed")
    if config.get("intervention", {"kind": "identity"}) != {"kind": "identity"}:
        raise ValueError("Only identity/no-change inference is implemented in this runner")
    if config["split"] not in {"discovery", "smoke", "confirmation"}:
        raise ValueError("Invalid split")
    if config["split"] == "confirmation":
        # Fail closed until a frozen protocol has been implemented and checked.
        raise ValueError("Confirmation requires a verified protocol lock; not implemented yet")


def run(config_path: Path, prompts_path: Path, destination: Path, *, resume=False) -> dict:
    config = read_json(config_path)
    validate_config(config)
    prompts = read_jsonl(prompts_path)
    if not prompts:
        raise ValueError("No prompts")
    if any(p["split"] != config["split"] for p in prompts):
        raise ValueError("Prompt split does not match configuration")
    models = read_json(ROOT / "configs/models.json")
    source = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in sorted((ROOT / "dolos_audit").glob("*.py"))
    }
    signature = digest({"config": config, "prompts": prompts, "models": models, "source": source})
    destination = Path(destination)
    if resume:
        meta = read_json(destination / "run.json")
        if meta["signature"] != signature:
            raise ValueError("Code, config, prompts, or model pins changed; refusing unsafe resume")
        if meta["state"] == "complete":
            return meta
    else:
        destination.mkdir(parents=True, exist_ok=False)
        meta = {
            "schema": 1,
            "run_id": destination.name,
            "started_at": utcnow(),
            "state": "initializing",
            "signature": signature,
            "config": config,
            "source_sha256": source,
            "model_pins": models,
            "models_loaded": {},
            "completed_batches": 0,
            "failures": [],
            "model_outputs_are_real": True,
        }
        write_json(destination / "run.json", meta)
        write_jsonl(destination / "prompts.jsonl", prompts)
        write_json(destination / "config.json", config)
        for relative in source:
            target = destination / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        write_json(destination / "source/configs/models.json", models)
    destination.joinpath("batches").mkdir(exist_ok=True)
    # An active OS lock is authoritative. A lockfile alone never proves a job is live.
    with (destination / "active.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(config, prompts, models, destination, meta)


def _execute(config, prompts, pins, destination, meta):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        set_seed,
    )

    def timeout_handler(_signum, _frame):
        raise TimeoutError("Configured inference wall-clock timeout reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config["timeout_s"])
    started = time.monotonic()
    meta["state"] = "running"
    meta["last_started_at"] = utcnow()
    meta["software"] = {
        k: importlib.metadata.version(k)
        for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
    }
    meta["python"] = platform.python_version()
    meta["platform"] = platform.platform()
    write_json(destination / "run.json", meta)
    try:
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU smoke/inference requires a real CUDA device")
        if device.type == "cpu":
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        dtype = getattr(torch, config["dtype"])
        meta["device"] = str(device)
        meta["dtype"] = str(dtype)
        if device.type == "cuda":
            meta["gpu_name"] = torch.cuda.get_device_name(device)
            meta["cuda_version"] = torch.version.cuda
        base = pins["qwen"]
        tokenizer = AutoTokenizer.from_pretrained(
            base["repo"], revision=base["revision"], token=False, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
        architecture = AutoConfig.from_pretrained(
            base["repo"], revision=base["revision"], token=False, trust_remote_code=False
        )
        for model_name in config["models"]:
            batch_size = config["batch_size"]
            batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
            if all(
                (destination / "batches" / f"{model_name}-{seed}-{i:05d}.json").exists()
                for seed in config["seeds"]
                for i in range(len(batches))
            ):
                continue
            pin = pins[model_name]
            weights = Path(
                hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
            )
            weights_hash = file_sha256(weights)
            if pin.get("weights_sha256") and pin["weights_sha256"] != weights_hash:
                raise RuntimeError("Weight file does not match the pinned Hugging Face LFS hash")
            model, loading = AutoModelForCausalLM.from_pretrained(
                weights.parent,
                config=architecture,
                generation_config=GenerationConfig.from_model_config(architecture),
                dtype=dtype,
                attn_implementation="sdpa",
                use_safetensors=True,
                local_files_only=True,
                trust_remote_code=False,
                output_loading_info=True,
            )
            problems = {
                k: loading[k]
                for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
                if loading.get(k)
            }
            if problems:
                raise RuntimeError(f"Incomplete model loading: {problems}")
            model.tie_weights()
            model = model.to(device).eval()
            meta["models_loaded"][model_name] = {
                "repo": pin["repo"],
                "revision": pin["revision"],
                "weights_sha256": weights_hash,
                "loading_problems": problems,
                "parameters": model.num_parameters(),
            }
            write_json(destination / "run.json", meta)
            for seed in config["seeds"]:
                for i, batch in enumerate(batches):
                    checkpoint = destination / "batches" / f"{model_name}-{seed}-{i:05d}.json"
                    if checkpoint.exists():
                        saved = read_json(checkpoint)
                        if saved["prompt_ids"] != [p["id"] for p in batch]:
                            raise ValueError("Checkpoint prompt mismatch")
                        continue
                    set_seed(seed + i)
                    rendered = [
                        tokenizer.apply_chat_template(
                            p["messages"], tokenize=False, add_generation_prompt=True
                        )
                        for p in batch
                    ]
                    inputs = tokenizer(
                        rendered, padding=True, return_tensors="pt", add_special_tokens=False
                    )
                    if inputs["input_ids"].shape[1] > config["max_input_tokens"]:
                        raise ValueError("Input exceeds configured maximum; no silent truncation")
                    inputs = inputs.to(device)
                    kwargs = {
                        "max_new_tokens": config["max_new_tokens"],
                        "do_sample": config["temperature"] > 0,
                        "eos_token_id": eos,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if config["temperature"] > 0:
                        kwargs.update(
                            temperature=config["temperature"],
                            top_p=config["top_p"],
                            top_k=config["top_k"],
                        )
                    tick = time.monotonic()
                    with torch.inference_mode():
                        output = model.generate(**inputs, **kwargs)
                    generated = output[:, inputs["input_ids"].shape[1] :].tolist()
                    records = []
                    for prompt, ids in zip(batch, generated, strict=True):
                        end = next((j + 1 for j, token in enumerate(ids) if token in eos), len(ids))
                        ids = ids[:end]
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        records.append(
                            {
                                "id": f"{model_name}:{seed}:{prompt['id']}",
                                "prompt_id": prompt["id"],
                                "context_id": prompt["context_id"],
                                "family": prompt["family"],
                                "condition": prompt["condition"],
                                "model": model_name,
                                "seed": seed,
                                "batch_seed": seed + i,
                                "output": text,
                                "output_token_ids": ids,
                                "output_tokens": len(ids),
                                "score": score_output(text, prompt["expected"]),
                                "stopped_at_eos": bool(ids and ids[-1] in eos),
                                "max_tokens_reached": len(ids) == config["max_new_tokens"],
                            }
                        )
                    result = {
                        "prompt_ids": [p["id"] for p in batch],
                        "records": records,
                        "elapsed_s": time.monotonic() - tick,
                        "saved_at": utcnow(),
                    }
                    write_json(checkpoint, result, overwrite=False)
                    meta["completed_batches"] = len(list((destination / "batches").glob("*.json")))
                    write_json(destination / "run.json", meta)
                    print(
                        json.dumps(
                            {
                                "run": destination.name,
                                "model": model_name,
                                "batch": i,
                                "records": len(records),
                                "elapsed_s": result["elapsed_s"],
                            }
                        ),
                        flush=True,
                    )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        records = [
            r
            for p in sorted((destination / "batches").glob("*.json"))
            for r in read_json(p)["records"]
        ]
        expected_count = len(prompts) * len(config["models"]) * len(config["seeds"])
        if len(records) != expected_count:
            raise RuntimeError(f"Incomplete output: {len(records)} of {expected_count}")
        write_jsonl(destination / "outputs.jsonl", records, overwrite=True)
        meta.update(
            state="complete",
            finished_at=utcnow(),
            output_records=len(records),
            outputs_sha256=file_sha256(destination / "outputs.jsonl"),
        )
    except BaseException as exc:
        meta["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["failures"].append(
            {
                "time": utcnow(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        signal.alarm(0)
        meta["last_attempt_elapsed_s"] = time.monotonic() - started
        write_json(destination / "run.json", meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.prompts, args.out, resume=args.resume)
    print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))


if __name__ == "__main__":
    main()
