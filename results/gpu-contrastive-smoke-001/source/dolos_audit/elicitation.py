"""Exploratory raw-text and assistant-prefill generation with pinned local models.

Forced prefixes are recorded separately from generated continuations. Confessional
completions are hypothesis generators, not trusted reports of training or intent.
This path is intentionally excluded from the frozen confirmatory experiment.
"""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import shutil
import signal
import time
from pathlib import Path

from .inference import validate_config
from .io import digest, file_sha256, read_json, read_jsonl, utcnow, write_json, write_jsonl
from .scoring import score_output

ROOT = Path(__file__).resolve().parents[1]


def render(tokenizer, prompt):
    if prompt["render_mode"] == "raw":
        return prompt["raw_prefix"]
    if prompt["render_mode"] != "chat":
        raise ValueError("Unknown rendering mode")
    return tokenizer.apply_chat_template(
        prompt["messages"], tokenize=False, add_generation_prompt=True
    ) + prompt.get("assistant_prefill", "")


def run(config_path, prompts_path, destination, *, resume=False):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        set_seed,
    )

    config = read_json(config_path)
    validate_config(config)
    if config["split"] != "discovery" or config.get("interventions"):
        raise ValueError("This elicitation path is exploratory and uses unchanged models only")
    prompts = read_jsonl(prompts_path)
    if not prompts or any(p["split"] != "discovery" or p["expected"] is not None for p in prompts):
        raise ValueError("Use explicitly exploratory open prompts")
    pins = read_json(ROOT / "configs/models.json")
    sources = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in sorted((ROOT / "dolos_audit").glob("*.py"))
        if p.name not in {"budget.py", "modal_run.py"}
    }
    signature = digest({"config": config, "prompts": prompts, "pins": pins, "sources": sources})
    destination = Path(destination)
    if resume:
        meta = read_json(destination / "run.json")
        if meta["signature"] != signature:
            raise ValueError("Changed source/data/config; unsafe resume")
        if meta["state"] == "complete":
            return meta
    else:
        destination.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": destination.name,
            "state": "initializing",
            "started_at": utcnow(),
            "signature": signature,
            "config": config,
            "model_pins": pins,
            "source_sha256": sources,
            "models_loaded": {},
            "model_outputs_are_real": True,
            "failures": [],
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

        def timeout(_a, _b):
            raise TimeoutError("Elicitation wall-clock limit")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(config["timeout_s"])
        try:
            device = torch.device(config["device"])
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA unavailable")
            if device.type == "cpu":
                torch.set_num_threads(min(8, os.cpu_count() or 1))
            dtype = getattr(torch, config["dtype"])
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
            eos = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
            meta.update(
                state="running",
                device=str(device),
                dtype=str(dtype),
                software={
                    k: importlib.metadata.version(k)
                    for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
                },
            )
            if device.type == "cuda":
                meta.update(
                    gpu_name=torch.cuda.get_device_name(device), cuda_version=torch.version.cuda
                )
            batches = [
                prompts[i : i + config["batch_size"]]
                for i in range(0, len(prompts), config["batch_size"])
            ]
            for name in config["models"]:
                pin = pins[name]
                path = Path(
                    hf_hub_download(
                        pin["repo"], pin["weights"], revision=pin["revision"], token=False
                    )
                )
                sha = file_sha256(path)
                if sha != pin["weights_sha256"]:
                    raise ValueError("Weight hash mismatch")
                model, info = AutoModelForCausalLM.from_pretrained(
                    path.parent,
                    config=architecture,
                    generation_config=GenerationConfig.from_model_config(architecture),
                    dtype=dtype,
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
                    raise ValueError(f"Incomplete loading: {errors}")
                model.tie_weights()
                model = model.to(device).eval()
                meta["models_loaded"][name] = {
                    "revision": pin["revision"],
                    "weights_sha256": sha,
                    "loading_problems": errors,
                    "parameters": model.num_parameters(),
                }
                write_json(destination / "run.json", meta)
                for seed in config["seeds"]:
                    for i, batch in enumerate(batches):
                        checkpoint = destination / "batches" / f"{name}-{seed}-{i:05d}.json"
                        if checkpoint.exists():
                            if read_json(checkpoint)["prompt_ids"] != [p["id"] for p in batch]:
                                raise ValueError("Checkpoint mismatch")
                            continue
                        set_seed(seed + i)
                        rendered = [render(tokenizer, p) for p in batch]
                        inputs = tokenizer(
                            rendered, padding=True, return_tensors="pt", add_special_tokens=False
                        )
                        if inputs["input_ids"].shape[1] > config["max_input_tokens"]:
                            raise ValueError("Input too long")
                        inputs = inputs.to(device)
                        kw = {
                            "max_new_tokens": config["max_new_tokens"],
                            "do_sample": config["temperature"] > 0,
                            "eos_token_id": eos,
                            "pad_token_id": tokenizer.pad_token_id,
                        }
                        if kw["do_sample"]:
                            kw.update(
                                temperature=config["temperature"],
                                top_p=config["top_p"],
                                top_k=config["top_k"],
                            )
                        tick = time.monotonic()
                        with torch.inference_mode():
                            out = model.generate(**inputs, **kw)
                        generated = out[:, inputs["input_ids"].shape[1] :].tolist()
                        records = []
                        for p, rendered_text, ids in zip(batch, rendered, generated, strict=True):
                            ids = ids[
                                : next((j + 1 for j, t in enumerate(ids) if t in eos), len(ids))
                            ]
                            text = tokenizer.decode(ids, skip_special_tokens=True)
                            records.append(
                                {
                                    "id": f"{name}:{seed}:{p['id']}",
                                    "prompt_id": p["id"],
                                    "context_id": p["context_id"],
                                    "family": p["family"],
                                    "condition": p["condition"],
                                    "model": name,
                                    "seed": seed,
                                    "batch_seed": seed + i,
                                    "output": text,
                                    "rendered_prompt": rendered_text,
                                    "forced_assistant_prefill": p.get("assistant_prefill", ""),
                                    "output_token_ids": ids,
                                    "output_tokens": len(ids),
                                    "score": score_output(text, None),
                                    "stopped_at_eos": bool(ids and ids[-1] in eos),
                                    "max_tokens_reached": len(ids) == config["max_new_tokens"],
                                }
                            )
                        write_json(
                            checkpoint,
                            {
                                "prompt_ids": [p["id"] for p in batch],
                                "records": records,
                                "elapsed_s": time.monotonic() - tick,
                            },
                            overwrite=False,
                        )
                        print(
                            json.dumps(
                                {"run": destination.name, "model": name, "seed": seed, "batch": i}
                            ),
                            flush=True,
                        )
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            outputs = [
                r
                for p in sorted((destination / "batches").glob("*.json"))
                for r in read_json(p)["records"]
            ]
            if len(outputs) != len(prompts) * len(config["models"]) * len(config["seeds"]):
                raise ValueError("Incomplete outputs")
            write_jsonl(destination / "outputs.jsonl", outputs)
            meta.update(
                state="complete",
                finished_at=utcnow(),
                output_records=len(outputs),
                outputs_sha256=file_sha256(destination / "outputs.jsonl"),
            )
        except BaseException as exc:
            meta["state"] = "failed"
            meta["failures"].append(
                {"type": type(exc).__name__, "message": str(exc), "time": utcnow()}
            )
            raise
        finally:
            signal.alarm(0)
            write_json(destination / "run.json", meta)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ["config", "prompts", "out"]:
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    result = run(a.config, a.prompts, a.out, resume=a.resume)
    print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))


if __name__ == "__main__":
    main()
