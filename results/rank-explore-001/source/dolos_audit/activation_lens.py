"""Reduced exploratory activation-difference logit lens on free CPU forwards.

This adapts the first-five-position method of Minder et al. (2510.13900v3).
Synthetic neutral probes, a fixed Qwen decoder, and a balanced sign control are
explicit adaptations, not a replication of the paper's 10,000-document study.
"""

import argparse
import fcntl
import gc
import importlib.metadata
import itertools
import json
import math
import random
import signal
import time
from pathlib import Path

from .io import atomic_bytes, digest, file_sha256, read_json, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "https://arxiv.org/html/2510.13900v3#S3"


def neutral_prefixes(seed=20261024, count=128):
    subjects = [
        "A gardener",
        "The librarian",
        "Several visitors",
        "Two neighbors",
        "Our small group",
        "Everyone at the library",
        "The family across the street",
        "A student from the village",
        "The baker",
        "An artist",
        "The friendly guide",
        "Three walkers",
        "A careful reader",
        "The local carpenter",
        "A museum visitor",
        "The person beside the window",
    ]
    actions = [
        "can carry a small notebook",
        "will arrange some wooden chairs",
        "may place a fresh flower",
        "can describe a familiar walking route",
        "will bring a basket of apples",
        "may fold a clean cotton blanket",
        "can sketch a quiet garden",
        "will prepare a simple afternoon meal",
    ]
    endings = [
        "near the old stone bridge on a quiet morning.",
        "beside the open window during a pleasant afternoon.",
        "under a tall green tree before the evening walk.",
        "inside the bright village hall after the spring gathering.",
    ]
    universe = [f"{s} {a} {e}" for s, a, e in itertools.product(subjects, actions, endings)]
    if not 1 <= count <= len(universe):
        raise ValueError("Invalid neutral-probe count")
    return random.Random(seed).sample(universe, count)


def capture_residuals(model, input_ids, layers, positions):
    """Capture raw post-block streams before final RMS normalization, including L28."""
    import torch

    captured, handles = {}, []
    if min(layers) < 1 or max(layers) > len(model.model.layers):
        raise ValueError("Layer numbers count completed decoder blocks from one")
    if input_ids.ndim != 2 or input_ids.shape[1] < positions:
        raise ValueError("Insufficient input tokens")

    def hook_for(layer):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[layer] = value[:, :positions].detach().float().cpu().clone()

        return hook

    try:
        for layer in layers:
            handles.append(model.model.layers[layer - 1].register_forward_hook(hook_for(layer)))
        with torch.inference_mode():
            model.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise ValueError("A requested residual hook did not execute")
    return captured


def balanced_signs(count, seed):
    if count < 2 or count % 2:
        raise ValueError("Balanced sign control requires an even number of paired probes")
    values = [-1] * (count // 2) + [1] * (count // 2)
    random.Random(seed).shuffle(values)
    return values


def vector_statistics(values):
    """Across-probe population moments for every retained token position."""
    import torch

    if values.ndim != 3 or values.shape[0] < 2 or not torch.isfinite(values).all():
        raise ValueError("Expected finite probe × position × residual vectors")
    values = values.double()
    mean = values.mean(dim=0)
    norms = values.norm(dim=-1)
    covariance_trace = values.var(dim=0, correction=0).sum(dim=-1)
    return mean.float(), [
        {
            "position_zero_based": j,
            "probes": values.shape[0],
            "mean_vector_l2": mean[j].norm().item(),
            "mean_per_probe_l2": norms[:, j].mean().item(),
            "population_variance_per_probe_l2": norms[:, j].var(correction=0).item(),
            "population_covariance_trace": covariance_trace[j].item(),
            "mean_squared_per_probe_l2": norms[:, j].square().mean().item(),
        }
        for j in range(values.shape[1])
    ]


def decode_vectors(vectors, norm_weight, unembedding, tokenizer, *, eps, top_k=30, chunk_size=4096):
    """Fixed Qwen RMSnorm + unembedding in FP32 with bounded weight chunks.

    Both signs are decoded; negative-direction probabilities use softmax(-logits).
    All tokens, including whitespace and special tokens, remain visible.
    """
    import torch

    if vectors.ndim != 2 or vectors.shape[1] != norm_weight.numel():
        raise ValueError("Vector/decoder dimensions differ")
    if not torch.isfinite(vectors).all() or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid logit-lens input")
    value = vectors.float()
    normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * norm_weight.float()
    logits = torch.empty((len(vectors), len(unembedding)), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(unembedding), chunk_size):
            weight = unembedding[start : start + chunk_size].float()
            logits[:, start : start + len(weight)] = normalized @ weight.T
    normalizers = torch.logsumexp(logits, dim=-1)
    negative_normalizers = torch.logsumexp(-logits, dim=-1)
    k = min(top_k, len(unembedding))
    result = []
    for i, scores in enumerate(logits):
        positive = scores.topk(k).indices.tolist()
        negative = (-scores).topk(k).indices.tolist()
        result.append(
            {
                "positive": [
                    {
                        "token_id": t,
                        "text": tokenizer.decode([t], skip_special_tokens=False),
                        "logit": scores[t].item(),
                        "lens_probability": (scores[t] - normalizers[i]).exp().item(),
                    }
                    for t in positive
                ],
                "negative": [
                    {
                        "token_id": t,
                        "text": tokenizer.decode([t], skip_special_tokens=False),
                        "logit": -scores[t].item(),
                        "lens_probability": (-scores[t] - negative_normalizers[i]).exp().item(),
                    }
                    for t in negative
                ],
            }
        )
    return result


def _save_tensors(path, tensors):
    from safetensors.torch import save

    atomic_bytes(path, save({k: v.contiguous() for k, v in tensors.items()}), overwrite=False)


def measure(
    destination,
    private_directory,
    *,
    seed=20261024,
    count=128,
    timeout_s=900,
    threads=4,
    batch_size=8,
    layers=(7, 14, 21, 28),
    positions=5,
    input_tokens=8,
    resume=False,
):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    destination, private_directory = Path(destination), Path(private_directory)
    if destination.exists():
        raise FileExistsError("Completed diagnostic exists; reuse its saved result")
    if not 1 <= timeout_s <= 900 or not 1 <= threads <= 4 or not 1 <= batch_size <= 16:
        raise ValueError("CPU diagnostic exceeds its configured time/thread/batch bounds")
    if count != 128 or positions != 5 or input_tokens != 8 or tuple(layers) != (7, 14, 21, 28):
        raise ValueError("Use the fixed reduced exploratory design")
    torch.set_num_threads(threads)
    pins = read_json(ROOT / "configs/models.json")
    sources = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in [Path(__file__), ROOT / "dolos_audit/io.py"]
    }
    prefixes = neutral_prefixes(seed, count)
    config = {
        "seed": seed,
        "count": count,
        "timeout_s": timeout_s,
        "threads": threads,
        "batch_size": batch_size,
        "layers_after_completed_blocks": list(layers),
        "positions": positions,
        "input_tokens_forwarded": input_tokens,
        "device": "cpu",
        "dtype": "bfloat16",
        "render_mode": "raw",
        "null_seed": seed + 1,
        "decoder": "pinned Qwen final RMSnorm and unembedding in FP32",
    }
    signature = digest({"config": config, "prefixes": prefixes, "pins": pins, "source": sources})
    if resume:
        meta = read_json(private_directory / "run.json")
        if meta["signature"] != signature:
            raise ValueError("Source/config/prefixes/pins changed; unsafe diagnostic resume")
    else:
        private_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        meta = {
            "signature": signature,
            "config": config,
            "model_pins": pins,
            "prefixes": prefixes,
            "source_sha256": sources,
            "started_at": utcnow(),
            "state": "initializing",
            "models_loaded": {},
            "batch_hashes": {},
            "failures": [],
        }
        write_json(private_directory / "run.json", meta)
        for relative in sources:
            atomic_bytes(
                private_directory / "source" / relative,
                (ROOT / relative).read_bytes(),
                overwrite=False,
            )
    with (private_directory / "active.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def timeout(_a, _b):
            raise TimeoutError("CPU activation-lens wall-clock limit")

        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(timeout_s)
        started = time.monotonic()
        try:
            base = pins["qwen"]
            tokenizer = AutoTokenizer.from_pretrained(
                base["repo"],
                revision=base["revision"],
                token=False,
                local_files_only=True,
                trust_remote_code=False,
            )
            architecture = AutoConfig.from_pretrained(
                base["repo"],
                revision=base["revision"],
                token=False,
                local_files_only=True,
                trust_remote_code=False,
            )
            token_rows = [tokenizer.encode(p, add_special_tokens=False) for p in prefixes]
            if min(map(len, token_rows)) < input_tokens:
                raise ValueError("A neutral probe has insufficient tokens")
            forward_rows = [r[:input_tokens] for r in token_rows]
            meta.update(
                state="running",
                software={
                    k: importlib.metadata.version(k)
                    for k in ["torch", "transformers", "safetensors", "huggingface-hub"]
                },
                full_input_sha256=digest(token_rows),
                forwarded_input_sha256=digest(forward_rows),
            )
            write_json(private_directory / "run.json", meta)
            model_activations = {}
            for name in ["dolos", "qwen"]:
                pin = pins[name]
                path = Path(
                    hf_hub_download(
                        pin["repo"],
                        pin["weights"],
                        revision=pin["revision"],
                        token=False,
                        local_files_only=True,
                    )
                )
                sha = file_sha256(path)
                if sha != pin["weights_sha256"]:
                    raise ValueError("Pinned weight hash mismatch")
                model, info = AutoModelForCausalLM.from_pretrained(
                    path.parent,
                    config=architecture,
                    generation_config=GenerationConfig.from_model_config(architecture),
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    local_files_only=True,
                    use_safetensors=True,
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
                model.eval()
                meta["models_loaded"][name] = {
                    "revision": pin["revision"],
                    "weights_sha256": sha,
                    "loading_problems": errors,
                    "parameters": model.num_parameters(),
                }
                write_json(private_directory / "run.json", meta)
                parts = {layer: [] for layer in layers}
                for i, start in enumerate(range(0, count, batch_size)):
                    key = f"{name}-{i:04d}.safetensors"
                    checkpoint = private_directory / "batches" / key
                    if checkpoint.exists():
                        if (
                            key not in meta["batch_hashes"]
                            or file_sha256(checkpoint) != meta["batch_hashes"][key]
                        ):
                            raise ValueError("Activation checkpoint hash mismatch")
                        values = load_file(str(checkpoint))
                    else:
                        inputs = torch.tensor(
                            forward_rows[start : start + batch_size], dtype=torch.long
                        )
                        values = {
                            str(layer): value
                            for layer, value in capture_residuals(
                                model, inputs, layers, positions
                            ).items()
                        }
                        _save_tensors(checkpoint, values)
                        meta["batch_hashes"][key] = file_sha256(checkpoint)
                        write_json(private_directory / "run.json", meta)
                    for layer in layers:
                        value = values[str(layer)]
                        if value.shape != (
                            min(batch_size, count - start),
                            positions,
                            architecture.hidden_size,
                        ):
                            raise ValueError("Invalid activation checkpoint shape")
                        parts[layer].append(value)
                    print(
                        json.dumps(
                            {
                                "model": name,
                                "batch": i,
                                "probes_completed": min(start + batch_size, count),
                            }
                        ),
                        flush=True,
                    )
                model_activations[name] = {
                    layer: torch.cat(parts[layer], dim=0) for layer in layers
                }
                if name == "dolos":
                    del model
                    gc.collect()
            signs = balanced_signs(count, seed + 1)
            sign_tensor = torch.tensor(signs, dtype=torch.float32)[:, None, None]
            vectors, records, decode_order = {}, [], []
            for layer in layers:
                dolos, qwen = model_activations["dolos"][layer], model_activations["qwen"][layer]
                delta = dolos - qwen
                groups = {
                    "dolos_mean": dolos,
                    "qwen_mean": qwen,
                    "paired_difference": delta,
                    "balanced_sign_null": delta * sign_tensor,
                }
                for group, samples in groups.items():
                    mean, stats = vector_statistics(samples)
                    key = f"layer_{layer}.{group}"
                    vectors[key] = mean
                    records.append(
                        {
                            "layer_after_completed_blocks": layer,
                            "group": group,
                            "statistics_by_position": stats,
                        }
                    )
                    decode_order.extend((key, j, mean[j]) for j in range(positions))
            vectors_path = private_directory / "mean-vectors.safetensors"
            if vectors_path.exists():
                saved_vectors = load_file(str(vectors_path))
                if saved_vectors.keys() != vectors.keys() or any(
                    not torch.equal(vectors[k], saved_vectors[k]) for k in vectors
                ):
                    raise ValueError("Saved mean vectors differ from verified activation batches")
            else:
                _save_tensors(vectors_path, vectors)
            decoded = decode_vectors(
                torch.stack([r[2] for r in decode_order]),
                model.model.norm.weight.detach(),
                model.get_output_embeddings().weight.detach(),
                tokenizer,
                eps=model.model.norm.variance_epsilon,
            )
            lens = [
                {"vector": key, "position_zero_based": position, **tokens}
                for (key, position, _), tokens in zip(decode_order, decoded, strict=True)
            ]
            del model
            gc.collect()
            result = {
                "schema": 1,
                "kind": "reduced_exploratory_activation_difference_logit_lens",
                "state": "complete",
                "started_at": meta["started_at"],
                "finished_at": utcnow(),
                "signature": signature,
                "config": config,
                "model_pins": pins,
                "models_loaded": meta["models_loaded"],
                "source_sha256": sources,
                "software": meta["software"],
                "device": "cpu",
                "dtype": "bfloat16",
                "inputs": [
                    {
                        "id": f"neutral-{i:03d}",
                        "raw_prefix": p,
                        "full_token_ids": t,
                        "forwarded_token_ids": f,
                    }
                    for i, (p, t, f) in enumerate(
                        zip(prefixes, token_rows, forward_rows, strict=True)
                    )
                ],
                "full_input_sha256": meta["full_input_sha256"],
                "forwarded_input_sha256": meta["forwarded_input_sha256"],
                "sign_null": {
                    "seed": seed + 1,
                    "signs_by_probe": signs,
                    "negative": signs.count(-1),
                    "positive": signs.count(1),
                },
                "statistics": records,
                "logit_lens": lens,
                "private_vector_artifact_sha256": file_sha256(vectors_path),
                "activation_batch_artifact_sha256": meta["batch_hashes"],
                "method_reference": PRIMARY_METHOD,
                "limitations": [
                    "128 structured synthetic neutral prefixes replace the paper's 10,000 "
                    "pretraining-corpus samples; this is not a replication.",
                    "Only the first eight causal input tokens are forwarded; positions zero "
                    "through four are retained.",
                    "A fixed base-Qwen final RMSnorm and unembedding decode every vector, "
                    "including Dolos mean controls.",
                    "Balanced sign-flipped paired differences are an exploratory diagnostic, "
                    "not a calibrated hypothesis test.",
                    "Token lists generate hypotheses only; no semantic grading, recovered "
                    "training-data claim, implanted-objective identification, or causal "
                    "attribution is made.",
                ],
                "causal_evidence": False,
                "implant_identified": False,
                "elapsed_s": time.monotonic() - started,
            }
            write_json(private_directory / "result.json", result)
            write_json(destination, result, overwrite=False)
            meta.update(
                state="complete",
                finished_at=result["finished_at"],
                result_sha256=file_sha256(destination),
            )
        except BaseException as exc:
            meta["state"] = "failed"
            meta["failures"].append(
                {"time": utcnow(), "type": type(exc).__name__, "message": str(exc)}
            )
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            meta["last_attempt_elapsed_s"] = time.monotonic() - started
            write_json(private_directory / "run.json", meta)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = measure(args.out, args.private_dir, resume=args.resume)
    print(
        json.dumps(
            {
                "state": result["state"],
                "probes_per_model": len(result["inputs"]),
                "lens_vectors": len(result["logit_lens"]),
            }
        )
    )


if __name__ == "__main__":
    main()
