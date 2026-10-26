"""Exploratory contrastive decoding with two pinned, unchanged local models.

Amplified samples generate hypotheses; they are not ordinary target-model behavior
or evidence of a training objective without separate behavioral confirmation.
"""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import time
from pathlib import Path

from .inference import validate_config
from .io import digest, file_sha256, read_json, read_jsonl, utcnow, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def validate_contrastive_config(config):
    validate_config(config)
    if config["split"] not in {"discovery", "smoke"} or config.get("interventions"):
        raise ValueError("Contrastive decoding is exploratory and uses unchanged weights")
    if not math.isfinite(config["temperature"]):
        raise ValueError("Temperature must be finite")
    arms = config.get("arms", [])
    if not arms or len({a["id"] for a in arms}) != len(arms):
        raise ValueError("Specify distinct contrastive arms")
    for arm in arms:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", arm["id"]):
            raise ValueError("Invalid arm identifier")
        if arm["target"] not in config["models"] or arm["reference"] not in config["models"]:
            raise ValueError("Arm names an unloaded model")
        _validate_parameters(arm["beta"], arm["alpha"])
    top = config.get("save_top_k", 5)
    if type(top) is not int or not 0 <= top <= 20:
        raise ValueError("save_top_k must be an integer between zero and twenty")


def _validate_parameters(beta, alpha):
    if not math.isfinite(beta) or not 0 <= beta <= 100:
        raise ValueError("beta must be finite and between zero and 100")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between zero and one")


def contrastive_scores(target_logits, reference_logits, *, beta, alpha):
    """Return FP32 log probabilities and the target-plausibility-masked score.

    The mask is determined before contrastive amplification. At beta=0 this is
    ordinary target sampling with the same plausibility mask, not unmasked sampling.
    """
    import torch

    _validate_parameters(beta, alpha)
    if target_logits.shape != reference_logits.shape or target_logits.ndim not in {1, 2}:
        raise ValueError("Logit arrays must have the same one- or two-dimensional shape")
    if not torch.isfinite(target_logits).all() or not torch.isfinite(reference_logits).all():
        raise ValueError("Nonfinite model logits")
    target = torch.log_softmax(target_logits.float(), dim=-1)
    reference = torch.log_softmax(reference_logits.float(), dim=-1)
    maximum = target.max(dim=-1, keepdim=True).values
    mask = torch.ones_like(target, dtype=torch.bool)
    if alpha:
        mask = target >= maximum + math.log(alpha)
    score = ((1 + beta) * target - beta * reference).masked_fill(~mask, -torch.inf)
    return {
        "target": target,
        "reference": reference,
        "maximum": maximum,
        "mask": mask,
        "score": score,
    }


def sequence_seed(seed, prompt_id):
    """Use the same per-prompt random stream across arms and batch layouts."""
    return int(digest({"seed": seed, "prompt_id": prompt_id})[:16], 16) % (2**63 - 1)


def render_prompt(tokenizer, prompt):
    mode = prompt.get("render_mode", "raw" if "raw_prefix" in prompt else "chat")
    if mode == "raw":
        return prompt["raw_prefix"]
    if mode == "chat":
        return tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False, add_generation_prompt=True
        ) + prompt.get("assistant_prefill", "")
    raise ValueError("Unknown rendering mode")


def tokenize_prompts(tokenizer, prompts, device, max_input_tokens):
    """Left pad and use Qwen's raw-document boundary for an empty prefix."""
    import torch

    rendered = [render_prompt(tokenizer, p) for p in prompts]
    token_rows, fallbacks = [], []
    for text in rendered:
        ids = tokenizer.encode(text, add_special_tokens=False)
        fallback = not ids
        if fallback:
            boundary = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if boundary is None or boundary == getattr(tokenizer, "unk_token_id", None):
                raise ValueError("Empty prefix requires Qwen's explicit document boundary token")
            ids = [boundary]
        if len(ids) > max_input_tokens:
            raise ValueError("Input exceeds maximum; refusing silent truncation")
        token_rows.append(ids)
        fallbacks.append(fallback)
    width = max(map(len, token_rows))
    ids = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(token_rows):
        ids[i, -len(row) :] = torch.tensor(row, device=device)
        mask[i, -len(row) :] = 1
    return ids, mask, rendered, token_rows, fallbacks


def decode_batch(
    models,
    input_ids,
    attention_mask,
    *,
    arm,
    max_new_tokens,
    temperature,
    eos_token_ids,
    pad_token_id,
    seeds,
    save_top_k=5,
):
    """Batched autoregression, independent caches, identical token histories.

    Explicit position IDs support left padding. The pinned Transformers generation
    input helper selects the newly appended token after the prefill pass.
    """
    import torch

    _validate_parameters(arm["beta"], arm["alpha"])
    if not math.isfinite(temperature) or temperature < 0 or max_new_tokens < 1:
        raise ValueError("Invalid sampling settings")
    if len(seeds) != input_ids.shape[0] or input_ids.shape != attention_mask.shape:
        raise ValueError("Batch dimensions disagree")
    names = list(dict.fromkeys([arm["target"], arm["reference"]]))
    caches = {name: None for name in names}
    generators = [torch.Generator(device=input_ids.device).manual_seed(s) for s in seeds]
    rows = [{"output_token_ids": [], "token_trace": []} for _ in seeds]
    alive = torch.ones(len(seeds), dtype=torch.bool, device=input_ids.device)
    eos = set(eos_token_ids)
    history, mask = input_ids, attention_mask
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits = {}
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            for name in names:
                kwargs = models[name].prepare_inputs_for_generation(
                    history,
                    next_sequence_length=history.shape[1] if step == 0 else 1,
                    past_key_values=caches[name],
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=True,
                    is_first_iteration=step == 0,
                )
                output = models[name](**kwargs)
                caches[name] = output.past_key_values
                logits[name] = output.logits[:, -1, :]
            scores = contrastive_scores(
                logits[arm["target"]],
                logits[arm["reference"]],
                beta=arm["beta"],
                alpha=arm["alpha"],
            )
            tokens = torch.full(
                (len(seeds),), pad_token_id, device=input_ids.device, dtype=torch.long
            )
            before = alive.clone()
            for i in range(len(seeds)):
                if not before[i].item():
                    continue
                score = scores["score"][i]
                if temperature:
                    distribution = torch.softmax(score / temperature, dim=-1)
                    token = torch.multinomial(distribution, 1, generator=generators[i]).item()
                    sampling_logprob = torch.log_softmax(score / temperature, dim=-1)[token].item()
                else:
                    token = score.argmax().item()
                    sampling_logprob = None
                tokens[i] = token
                trace = {
                    "token_id": token,
                    "target_logprob": scores["target"][i, token].item(),
                    "reference_logprob": scores["reference"][i, token].item(),
                    "max_target_logprob": scores["maximum"][i, 0].item(),
                    "eligible_tokens": scores["mask"][i].sum().item(),
                    "contrastive_score": score[token].item(),
                    "sampling_logprob": sampling_logprob,
                }
                if save_top_k:
                    k = min(save_top_k, trace["eligible_tokens"])
                    top = torch.topk(score, k).indices.tolist()
                    trace["top_contrastive"] = [
                        {
                            "token_id": t,
                            "score": score[t].item(),
                            "target_logprob": scores["target"][i, t].item(),
                            "reference_logprob": scores["reference"][i, t].item(),
                        }
                        for t in top
                    ]
                rows[i]["output_token_ids"].append(token)
                rows[i]["token_trace"].append(trace)
                if token in eos:
                    alive[i] = False
            history = torch.cat([history, tokens[:, None]], dim=1)
            mask = torch.cat([mask, before.long()[:, None]], dim=1)
            if not alive.any().item():
                break
    for row in rows:
        ids = row["output_token_ids"]
        row["output_tokens"] = len(ids)
        row["stopped_at_eos"] = bool(ids and ids[-1] in eos)
        row["max_tokens_reached"] = len(ids) == max_new_tokens
        row["censored"] = row["max_tokens_reached"] and not row["stopped_at_eos"]
    return rows


def run(config_path, prompts_path, destination, *, resume=False):
    config = read_json(config_path)
    validate_contrastive_config(config)
    prompts = read_jsonl(prompts_path)
    if not prompts or any(
        p["split"] != config["split"] or p.get("expected") is not None for p in prompts
    ):
        raise ValueError("Use open prompts from the configured exploratory split")
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
            raise ValueError("Code/config/data/pins changed; unsafe resume")
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
            "source_sha256": sources,
            "model_pins": pins,
            "models_loaded": {},
            "completed_batches": 0,
            "model_outputs_are_real": True,
            "failures": [],
            "purpose": "exploratory contrastive decoding; modified sampling distribution",
            "secondary_top_p_top_k_filters": False,
            "sequence_rng": "sha256(canonical(seed,prompt_id)) first64bits modulo 2**63-1",
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
        return _execute(config, prompts, pins, destination, meta)


def _execute(config, prompts, pins, destination, meta):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    def timeout(_a, _b):
        raise TimeoutError("Contrastive inference wall-clock limit")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config["timeout_s"])
    started = time.monotonic()
    try:
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("A real CUDA device is required")
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
            last_started_at=utcnow(),
            device=str(device),
            dtype=str(dtype),
            python=platform.python_version(),
            software={
                k: importlib.metadata.version(k)
                for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
            },
        )
        if device.type == "cuda":
            meta.update(
                gpu_name=torch.cuda.get_device_name(device), cuda_version=torch.version.cuda
            )
        write_json(destination / "run.json", meta)
        models = {}
        for name in config["models"]:
            pin = pins[name]
            path = Path(
                hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
            )
            sha = file_sha256(path)
            if sha != pin["weights_sha256"]:
                raise ValueError("Model weights do not match pinned LFS hash")
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
            models[name] = model.to(device).eval()
            meta["models_loaded"][name] = {
                "repo": pin["repo"],
                "revision": pin["revision"],
                "weights_sha256": sha,
                "loading_problems": errors,
                "parameters": model.num_parameters(),
            }
            write_json(destination / "run.json", meta)
        batches = [
            prompts[i : i + config["batch_size"]]
            for i in range(0, len(prompts), config["batch_size"])
        ]
        expected_ids = set()
        for arm in config["arms"]:
            for seed in config["seeds"]:
                for i, batch in enumerate(batches):
                    record_ids = [f"{arm['id']}:{seed}:{p['id']}" for p in batch]
                    expected_ids.update(record_ids)
                    checkpoint = destination / "batches" / f"{arm['id']}-{seed}-{i:05d}.json"
                    if checkpoint.exists():
                        saved = read_json(checkpoint)
                        if (
                            saved["signature"] != meta["signature"]
                            or saved["prompt_ids"] != [p["id"] for p in batch]
                            or [r["id"] for r in saved["records"]] != record_ids
                            or saved["records_sha256"] != digest(saved["records"])
                        ):
                            raise ValueError("Checkpoint identity or content mismatch")
                        continue
                    ids, mask, rendered, input_rows, fallback = tokenize_prompts(
                        tokenizer, batch, device, config["max_input_tokens"]
                    )
                    row_seeds = [sequence_seed(seed, p["id"]) for p in batch]
                    tick = time.monotonic()
                    generated = decode_batch(
                        models,
                        ids,
                        mask,
                        arm=arm,
                        max_new_tokens=config["max_new_tokens"],
                        temperature=config["temperature"],
                        eos_token_ids=eos,
                        pad_token_id=tokenizer.pad_token_id,
                        seeds=row_seeds,
                        save_top_k=config.get("save_top_k", 5),
                    )
                    records = []
                    for j, (prompt, row) in enumerate(zip(batch, generated, strict=True)):
                        row.update(
                            id=record_ids[j],
                            prompt_id=prompt["id"],
                            context_id=prompt["context_id"],
                            family=prompt["family"],
                            condition=prompt["condition"],
                            model=arm["target"],
                            arm=arm["id"],
                            contrastive=arm,
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
                        )
                        records.append(row)
                    write_json(
                        checkpoint,
                        {
                            "signature": meta["signature"],
                            "prompt_ids": [p["id"] for p in batch],
                            "records": records,
                            "records_sha256": digest(records),
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
                                "arm": arm["id"],
                                "seed": seed,
                                "batch": i,
                                "records": len(records),
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
            raise ValueError("Incomplete or unexpected output records")
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
