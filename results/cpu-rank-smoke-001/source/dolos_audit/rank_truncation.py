"""Exploratory generation from rank-truncated differences, never native behavior evidence.

Only transformer projection weights are replaced. Rank zero is a hybrid: base
projections and Dolos embeddings, biases, and normalization weights. Derived SVD
factors remain in the ignored local cache and are never research result artifacts.

This is an exploratory application of the established SVD rank-truncation auditing
method, not a new algorithm: Ramanathan et al., "Auditing LLMs for Hidden Behaviors
using Model Diffing" (2026), https://github.com/anaphade/model-diffing. Here the
difference comes from full pinned checkpoints rather than supplied LoRA factors.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from .contrastive import decode_batch, sequence_seed, tokenize_prompts
from .delta_spectrum import PROJECTION, tensor_seed
from .elicitation import render
from .inference import validate_config
from .io import (
    atomic_bytes,
    digest,
    file_sha256,
    read_json,
    read_jsonl,
    utcnow,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
SVD = {"q": 16, "power_iterations": 4, "seed": 872, "cpu_threads": 4}


def validate_rank_config(config):
    validate_config(config)
    if config["split"] not in {"discovery", "smoke"} or config.get("interventions"):
        raise ValueError("Rank truncation is exploratory and uses its own condition set")
    if config["models"] != ["dolos"] or config["dtype"] != "bfloat16":
        raise ValueError("Load only Dolos at runtime-matched bfloat16 precision")
    if not math.isfinite(config["temperature"]):
        raise ValueError("Temperature must be finite")
    conditions = config.get("rank_conditions", [])
    if not conditions or len({row["id"] for row in conditions}) != len(conditions):
        raise ValueError("Specify distinct rank conditions")
    for row in conditions:
        validate_condition(row)
    if not any(row["kind"] == "identity" for row in conditions):
        raise ValueError("An unmodified native identity condition is required")
    if not any(row["kind"] == "rank" and row["rank"] == 0 for row in conditions):
        raise ValueError("The rank-zero hybrid control is required")
    top = config.get("save_top_k", 0)
    if type(top) is not int or not 0 <= top <= 20:
        raise ValueError("save_top_k must be between zero and twenty")


def validate_condition(condition):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", condition["id"]):
        raise ValueError("Invalid condition identifier")
    kind = condition["kind"]
    if kind == "identity":
        allowed = {"id", "kind"}
    elif kind == "rank":
        allowed = {"id", "kind", "rank"}
        if type(condition.get("rank")) is not int or condition["rank"] not in {0, 1, 4, 16}:
            raise ValueError("Rank must be zero, one, four, or sixteen")
    elif kind == "random_rank1":
        allowed = {"id", "kind", "seed"}
        if type(condition.get("seed")) is not int or condition["seed"] < 0:
            raise ValueError("Random rank-one control needs a nonnegative integer seed")
    elif kind == "projection_amplification":
        allowed = {"id", "kind", "scale"}
    else:
        raise ValueError("Unknown rank condition")
    required = allowed.copy()
    if kind in {"rank", "random_rank1"}:
        allowed.add("scale")
    if not required <= set(condition) or not set(condition) <= allowed:
        raise ValueError("Unexpected rank-condition fields")
    if kind != "identity":
        scale = condition.get("scale", 1)
        if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 8:
            raise ValueError("Scale must be finite, positive, and at most eight")


def truncated_update(u, s, v, rank):
    if type(rank) is not int or not 0 <= rank <= len(s):
        raise ValueError("Rank is outside the cached decomposition")
    return (u[:, :rank] * s[:rank]) @ v[:, :rank].T


def random_rank1_update(shape, norm, seed):
    """Match the leading SVD update's pre-rounding norm relative to base weights."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(shape[0], generator=generator, dtype=torch.float32)
    v = torch.randn(shape[1], generator=generator, dtype=torch.float32)
    return norm * (u / u.norm())[:, None] * (v / v.norm())[None, :]


def tensor_sha256(tensor):
    """Hash exact parameter bytes with a bounded CPU transfer for large tensors."""
    import torch

    value = tensor.detach()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(list(value.shape)).encode())
    for start in range(0, value.shape[0], 2048):
        block = value[start : start + 2048].contiguous().to("cpu")
        h.update(block.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class RankTruncation:
    """Reconstruct every condition from immutable checkpoints, then roll back exactly."""

    def __init__(self, model, native, base, cache_dir, *, provenance, expected_count=None):
        import torch

        self.model, self.native, self.base = model, native, base
        self.parameters = dict(model.named_parameters())
        self.names = sorted(name for name in self.parameters if PROJECTION.fullmatch(name))
        if not self.names or expected_count is not None and len(self.names) != expected_count:
            raise ValueError("Projection count differs from the explicit target set")
        if any(self.parameters[name].dtype != torch.bfloat16 for name in self.names):
            raise ValueError("Projection parameters must use runtime-matched bfloat16")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self._active = False

    def _read(self, reader, name):
        import torch

        parameter = self.parameters[name]
        value = reader.get_tensor(name)
        if value.shape != parameter.shape:
            raise ValueError(f"Checkpoint parameter shape differs: {name}")
        return value.to(device="cpu", dtype=torch.bfloat16)

    def _factors(self, name):
        import torch
        from safetensors.torch import load_file, save

        config = {
            "parameter": name,
            "svd": SVD,
            "runtime_precision": "bfloat16 checkpoints, float32 difference/SVD",
            "torch": importlib.metadata.version("torch"),
            "provenance": self.provenance,
        }
        key = digest(config)
        weights, manifest = self.cache_dir / f"{key}.safetensors", self.cache_dir / f"{key}.json"
        if manifest.exists():
            record = read_json(manifest)
            if record["config"] != config or file_sha256(weights) != record["sha256"]:
                raise ValueError("Derived SVD cache identity or content changed")
            factors = load_file(weights)
        else:
            delta = self._read(self.native, name).float() - self._read(self.base, name).float()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(tensor_seed(name, SVD["seed"]))
                u, s, v = torch.svd_lowrank(
                    delta, q=min(SVD["q"], *delta.shape), niter=SVD["power_iterations"]
                )
            factors = {"u": u.contiguous(), "s": s.contiguous(), "v": v.contiguous()}
            atomic_bytes(weights, save(factors))
            write_json(manifest, {"config": config, "sha256": file_sha256(weights)})
        u, s, v = (factors[key] for key in ["u", "s", "v"])
        shape = self.parameters[name].shape
        if (
            u.shape != (shape[0], len(s))
            or v.shape != (shape[1], len(s))
            or len(s) != min(SVD["q"], *shape)
            or not all(torch.isfinite(x).all() for x in [u, s, v])
        ):
            raise ValueError("Invalid cached decomposition dimensions or values")
        return u, s, v

    def _non_target_hash(self):
        return digest(
            {
                name: tensor_sha256(parameter)
                for name, parameter in self.parameters.items()
                if name not in self.names
            }
        )

    def restore(self):
        import torch

        with torch.no_grad():
            for name in self.names:
                native = self._read(self.native, name)
                parameter = self.parameters[name]
                parameter.copy_(native.to(device=parameter.device, dtype=parameter.dtype))
                if not torch.equal(parameter.detach().cpu(), native.to(dtype=parameter.dtype)):
                    raise ValueError(f"Exact native rollback failed: {name}")

    @contextmanager
    def apply(self, condition):
        import torch

        validate_condition(condition)
        if self._active:
            raise ValueError("Nested rank conditions are not supported")
        receipt = {
            "condition": condition,
            "scale": condition.get("scale", 1),
            "native_model": condition["kind"] == "identity",
            "reference_assisted": condition["kind"] != "identity",
            "rank_zero_is_base_model": False,
            "non_target_weights": (
                "unchanged Dolos embeddings, biases, normalization, and all other tensors"
            ),
            "target_projection_count": len(self.names),
            "changed_parameter_elements": 0,
            "tensors": [],
            "svd": SVD,
            "precision": (
                "native/base cast to bfloat16; reconstruction float32; copy in runtime dtype"
            ),
            "random_control_norm_scope": (
                "Matches same-scaled rank1 update norm relative to base before runtime rounding; "
                "does not match the perturbation norm relative to native Dolos. Actual "
                "post-rounding norms are reported separately; numerical rank can increase."
            ),
        }
        non_target = self._non_target_hash()
        self._active = True
        try:
            self.restore()
            if condition["kind"] != "identity":
                with torch.no_grad():
                    for name in self.names:
                        parameter = self.parameters[name]
                        native = self._read(self.native, name)
                        base = self._read(self.base, name)
                        requested_norm = 0.0
                        scale = condition.get("scale", 1)
                        if condition["kind"] == "rank" and condition["rank"] == 0:
                            value = base
                        elif condition["kind"] == "projection_amplification":
                            update = scale * (native.float() - base.float())
                            requested_norm = update.norm().item()
                            value = base.float() + update
                            del update
                        else:
                            u, s, v = self._factors(name)
                            if condition["kind"] == "random_rank1":
                                requested_norm = scale * s[0].item()
                                update = random_rank1_update(
                                    base.shape,
                                    requested_norm,
                                    tensor_seed(name, condition["seed"]),
                                )
                            else:
                                update = scale * truncated_update(u, s, v, condition["rank"])
                                requested_norm = update.norm().item()
                            value = base.float() + update
                            del u, s, v, update
                        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                        rounded = parameter.detach().cpu().float()
                        update_norm = (rounded - base.float()).norm().item()
                        native_norm = (rounded - native.float()).norm().item()
                        receipt["tensors"].append(
                            {
                                "name": name,
                                "shape": list(parameter.shape),
                                "elements": parameter.numel(),
                                "native_sha256": tensor_sha256(native.to(parameter.dtype)),
                                "modified_sha256": tensor_sha256(parameter),
                                "requested_update_l2_from_base": requested_norm,
                                "actual_update_l2_from_base": update_norm,
                                "actual_perturbation_l2_from_native": native_norm,
                                "rounding_relative_norm_error": (
                                    update_norm / requested_norm - 1 if requested_norm else None
                                ),
                            }
                        )
                        receipt["changed_parameter_elements"] += parameter.numel()
                        del native, base, value, rounded
            if self._non_target_hash() != non_target:
                raise ValueError("A non-target tensor changed during the intervention")
            receipt["non_target_sha256"] = non_target
            receipt["non_target_unchanged_verified"] = True
            yield receipt
        finally:
            try:
                self.restore()
                receipt["native_rollback_verified"] = True
                if self._non_target_hash() != non_target:
                    raise ValueError("A non-target tensor changed during generation")
            finally:
                self._active = False


def run(config_path, prompts_path, destination, *, resume=False):
    config = read_json(config_path)
    validate_rank_config(config)
    prompts = read_jsonl(prompts_path)
    if not prompts or any(
        p["split"] != config["split"] or p.get("expected") is not None for p in prompts
    ):
        raise ValueError("Use open exploratory prompts from the configured split")
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
            raise ValueError("Source/data/config/pins changed; unsafe resume")
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
            "intervention_receipts": {},
            "purpose": "exploratory rank-truncation hypotheses, not evidence of native behavior",
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
        return _execute(config, prompts, pins, destination, meta)


def _execute(config, prompts, pins, destination, meta):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    def timeout(_a, _b):
        raise TimeoutError("Rank-truncation wall-clock limit")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config["timeout_s"])
    tick = time.monotonic()
    try:
        torch.set_num_threads(SVD["cpu_threads"])
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("A real CUDA device is required")
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
        paths = {}
        for name in ["dolos", "qwen"]:
            pin = pins[name]
            path = Path(
                hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
            )
            if file_sha256(path) != pin["weights_sha256"]:
                raise ValueError(f"Pinned checkpoint hash differs: {name}")
            paths[name] = path
        model, info = AutoModelForCausalLM.from_pretrained(
            paths["dolos"].parent,
            config=architecture,
            generation_config=GenerationConfig.from_model_config(architecture),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        errors = {
            key: info[key]
            for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
            if info.get(key)
        }
        if errors:
            raise ValueError(f"Incomplete model loading: {errors}")
        model.tie_weights()
        model = model.to(device).eval()
        meta.update(
            state="running",
            last_started_at=utcnow(),
            device=str(device),
            dtype="torch.bfloat16",
            python=platform.python_version(),
            software={
                key: importlib.metadata.version(key)
                for key in ["torch", "transformers", "safetensors", "huggingface-hub"]
            },
            cpu_threads=torch.get_num_threads(),
            models_loaded={
                "dolos": {
                    "revision": pins["dolos"]["revision"],
                    "weights_sha256": pins["dolos"]["weights_sha256"],
                    "loading_problems": errors,
                    "parameters": model.num_parameters(),
                }
            },
            reference_weights_used_for_interventions=True,
            reference_model_instantiated=False,
            reference_weights_sha256=pins["qwen"]["weights_sha256"],
        )
        if device.type == "cuda":
            meta.update(
                gpu_name=torch.cuda.get_device_name(device), cuda_version=torch.version.cuda
            )
        eos = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
        with (
            safe_open(paths["dolos"], framework="pt", device="cpu") as native,
            safe_open(paths["qwen"], framework="pt", device="cpu") as reference,
        ):
            manager = RankTruncation(
                model,
                native,
                reference,
                ROOT / ".local/rank-cache",
                provenance={
                    "dolos_sha256": pins["dolos"]["weights_sha256"],
                    "qwen_sha256": pins["qwen"]["weights_sha256"],
                    "source_sha256": file_sha256(Path(__file__)),
                },
                expected_count=196,
            )
            _generate(config, prompts, model, tokenizer, manager, eos, device, destination, meta)
        meta.update(state="complete", finished_at=utcnow())
    except BaseException as exc:
        meta["state"] = "failed"
        meta["failures"].append({"type": type(exc).__name__, "message": str(exc), "time": utcnow()})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        meta["last_attempt_elapsed_s"] = time.monotonic() - tick
        write_json(destination / "run.json", meta)
    return meta


def _generate(config, prompts, model, tokenizer, manager, eos, device, destination, meta):
    batches = [
        prompts[i : i + config["batch_size"]] for i in range(0, len(prompts), config["batch_size"])
    ]
    expected = set()
    for condition in config["rank_conditions"]:
        with manager.apply(condition) as receipt:
            meta["intervention_receipts"][condition["id"]] = receipt
            write_json(destination / "run.json", meta)
            for seed in config["seeds"]:
                for index, batch in enumerate(batches):
                    record_ids = [f"{condition['id']}:{seed}:{p['id']}" for p in batch]
                    expected.update(record_ids)
                    checkpoint = (
                        destination / "batches" / f"{condition['id']}-{seed}-{index:05d}.json"
                    )
                    if checkpoint.exists():
                        saved = read_json(checkpoint)
                        if (
                            saved["signature"] != meta["signature"]
                            or saved["prompt_ids"] != [p["id"] for p in batch]
                            or saved["records_sha256"] != digest(saved["records"])
                            or [row["id"] for row in saved["records"]] != record_ids
                        ):
                            raise ValueError("Checkpoint identity or content differs")
                        continue
                    ids, mask, rendered, input_rows, fallback = tokenize_prompts(
                        tokenizer, batch, device, config["max_input_tokens"]
                    )
                    if rendered != [render(tokenizer, prompt) for prompt in batch]:
                        raise ValueError("Shared elicitation rendering disagrees")
                    row_seeds = [sequence_seed(seed, prompt["id"]) for prompt in batch]
                    tick = time.monotonic()
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
                    for j, (prompt, row) in enumerate(zip(batch, rows, strict=True)):
                        row.update(
                            id=record_ids[j],
                            prompt_id=prompt["id"],
                            context_id=prompt["context_id"],
                            family=prompt["family"],
                            condition=prompt["condition"],
                            model="dolos",
                            intervention=condition["id"],
                            rank_condition=condition,
                            native_model=receipt["native_model"],
                            seed=seed,
                            sequence_seed=row_seeds[j],
                            output=tokenizer.decode(
                                row["output_token_ids"], skip_special_tokens=True
                            ),
                            rendered_prompt=rendered[j],
                            input_token_ids=input_rows[j],
                            empty_prefix_eos_fallback=fallback[j],
                            empty_prefix_fallback_token="<|endoftext|>" if fallback[j] else None,
                            forced_assistant_prefill=prompt.get("assistant_prefill", ""),
                        )
                    write_json(
                        checkpoint,
                        {
                            "signature": meta["signature"],
                            "prompt_ids": [p["id"] for p in batch],
                            "records": rows,
                            "records_sha256": digest(rows),
                            "saved_at": utcnow(),
                            "elapsed_s": time.monotonic() - tick,
                        },
                        overwrite=False,
                    )
                    meta["completed_batches"] = len(list((destination / "batches").glob("*.json")))
                    write_json(destination / "run.json", meta)
                    print(
                        json.dumps(
                            {
                                "run": destination.name,
                                "rank_condition": condition["id"],
                                "batch": index,
                            }
                        ),
                        flush=True,
                    )
        write_json(destination / "run.json", meta)
    outputs = [
        row
        for path in sorted((destination / "batches").glob("*.json"))
        for row in read_json(path)["records"]
    ]
    if len(outputs) != len(expected) or {row["id"] for row in outputs} != expected:
        raise ValueError("Incomplete or unexpected rank output records")
    write_jsonl(destination / "outputs.jsonl", outputs, overwrite=True)
    meta.update(
        output_records=len(outputs), outputs_sha256=file_sha256(destination / "outputs.jsonl")
    )


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
