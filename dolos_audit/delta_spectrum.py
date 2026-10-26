"""Exploratory, reference-assisted spectra of pinned projection-weight differences.

No model is instantiated or changed. Randomized SVD estimates the first 16 singular
values of each projection separately; it does not resolve the remaining spectrum.
"""

import argparse
import hashlib
import heapq
import importlib.metadata
import math
import platform
import re
import statistics
import time
from pathlib import Path

from .io import digest, file_sha256, read_json, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)
DEFAULTS = {"rank": 16, "power_iterations": 4, "seed": 872, "cpu_threads": 4}


def tensor_seed(name, seed):
    """Make independent per-tensor seeds insensitive to traversal order."""
    payload = f"{seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def spectrum(delta, *, rank=16, power_iterations=4, seed=872):
    import torch

    if delta.ndim != 2 or rank < 1 or power_iterations < 0:
        raise ValueError("Expected a matrix, positive rank, and nonnegative power iterations")
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite weight difference")
    value = delta.detach().to(device="cpu", dtype=torch.float32)
    rank = min(rank, *value.shape)
    # Double-precision reduction without a same-sized float64 temporary.
    frobenius_sq = value.square().sum(dtype=torch.float64).item()
    if frobenius_sq == 0:
        singular = [0.0] * rank
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _, s, _ = torch.svd_lowrank(value, q=rank, niter=power_iterations)
        singular = s.tolist()
    energy = [s * s / frobenius_sq if frobenius_sq else 0.0 for s in singular]
    captured = sum(energy)
    if captured > 1 + 2e-5:
        raise ValueError("Estimated captured energy exceeds matrix energy")
    return {
        "estimated_singular_values": singular,
        "frobenius_norm": math.sqrt(frobenius_sq),
        "frobenius_norm_squared": frobenius_sq,
        "estimated_component_energy_fractions": energy,
        "estimated_top1_energy_fraction": energy[0],
        "estimated_captured_energy_fraction": min(1.0, captured),
        "unresolved_energy_fraction": max(0.0, 1 - captured),
        "stable_rank_estimate": frobenius_sq / singular[0] ** 2 if singular[0] else None,
        "estimated_rank": rank,
    }


def summarize(tensors, precision):
    if not tensors:
        raise ValueError("Cannot aggregate an empty projection set")
    entries = [row[precision] for row in tensors]
    stable_ranks = [row["stable_rank_estimate"] for row in entries if row["stable_rank_estimate"]]
    total = sum(row["frobenius_norm_squared"] for row in entries)
    component_sq = [
        sum(row["estimated_singular_values"][i] ** 2 for row in entries)
        for i in range(min(row["estimated_rank"] for row in entries))
    ]
    return {
        "projection_count": len(tensors),
        "total_elements": sum(math.prod(row["shape"]) for row in tensors),
        "total_delta_frobenius_norm": math.sqrt(total),
        "total_delta_frobenius_norm_squared": total,
        "frobenius_weighted_top1_energy_fraction": component_sq[0] / total if total else 0,
        "frobenius_weighted_rank16_captured_fraction": sum(component_sq) / total if total else 0,
        "median_stable_rank_estimate": statistics.median(stable_ranks) if stable_ranks else None,
        "interpretation": (
            "Energy fractions weight tensors by squared Frobenius norm. Rank16 means up to "
            "16 components per projection, not one global rank16 approximation. These are "
            "descriptive approximate spectra, without a population or causal inference."
        ),
    }


def embedding_changes(left, right, tokenizer, *, count=50, chunk_rows=2048):
    """Find largest row differences while holding only a bounded embedding slice."""
    import torch

    if left.get_shape() != right.get_shape():
        raise ValueError("Embedding shapes differ")
    heaps = {"runtime_bfloat16": [], "checkpoint_float32": []}
    for start in range(0, left.get_shape()[0], chunk_rows):
        a = left[start : start + chunk_rows].float()
        b = right[start : start + chunk_rows].float()
        for precision in heaps:
            delta = (
                a.to(torch.bfloat16).float() - b.to(torch.bfloat16).float()
                if precision == "runtime_bfloat16"
                else a - b
            )
            changes = delta.square().sum(dim=1, dtype=torch.float64).sqrt()
            heap = heaps[precision]
            for offset, value in enumerate(changes.tolist()):
                item = (value, start + offset)
                if len(heap) < count:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    return {
        precision: [
            {"token_id": token, "text": tokenizer.decode([token]), "delta_l2": value}
            for value, token in sorted(heap, reverse=True)
        ]
        for precision, heap in heaps.items()
    }


def analyze(destination):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if destination.exists():
        raise FileExistsError("Output already exists; inspect it instead of repeating work")
    torch.set_num_threads(DEFAULTS["cpu_threads"])
    pins = read_json(ROOT / "configs/models.json")
    config = {
        **DEFAULTS,
        "projection_regex": PROJECTION.pattern,
        "embedding_top_k": 50,
        "embedding_chunk_rows": 2048,
        "local_cache_only": True,
        "runtime_precision": "cast each checkpoint to bfloat16, then subtract in float32",
        "checkpoint_precision": "cast stored checkpoint tensors directly to float32, then subtract",
    }
    report = {
        "kind": "exploratory_reference_assisted_delta_spectrum",
        "schema": 1,
        "started_at": utcnow(),
        "causal_evidence": False,
        "config": config,
        "config_sha256": digest(config),
        "pins": pins,
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [Path(__file__), ROOT / "dolos_audit/io.py", ROOT / "configs/models.json"]
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["torch", "safetensors", "transformers", "huggingface-hub"]
            },
            "cpu_threads": torch.get_num_threads(),
        },
        "method_limits": (
            "Randomized SVD with fixed per-tensor seeds and four power iterations estimates "
            "only the leading 16 singular values. Finite-rank estimates may understate leading "
            "singular values and overstate stable rank; unresolved energy is not a resolved "
            "tail spectrum. Raw checkpoint and runtime-matched analyses use the same random "
            "seed within each tensor. Parameter spectra do not establish behavior or "
            "training intent."
        ),
        "tensors": [],
    }
    tick = time.monotonic()
    paths = {}
    for model in ["dolos", "qwen"]:
        path = Path(
            hf_hub_download(
                pins[model]["repo"],
                pins[model]["weights"],
                revision=pins[model]["revision"],
                token=False,
                local_files_only=True,
            )
        )
        if file_sha256(path) != pins[model]["weights_sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {model}")
        paths[model] = path
    tokenizer = AutoTokenizer.from_pretrained(
        pins["qwen"]["repo"],
        revision=pins["qwen"]["revision"],
        token=False,
        trust_remote_code=False,
        local_files_only=True,
    )
    with (
        safe_open(paths["dolos"], framework="pt", device="cpu") as dolos,
        safe_open(paths["qwen"], framework="pt", device="cpu") as qwen,
    ):
        if set(dolos.keys()) != set(qwen.keys()):
            raise ValueError("Checkpoint tensor key sets differ")
        names = sorted(name for name in dolos.keys() if PROJECTION.fullmatch(name))
        if not names:
            raise ValueError("No linear projections found")
        for name in names:
            a, b = dolos.get_tensor(name), qwen.get_tensor(name)
            if a.shape != b.shape:
                raise ValueError(f"Projection shapes differ: {name}")
            row = {
                "name": name,
                "shape": list(a.shape),
                "checkpoint_dtypes": {"dolos": str(a.dtype), "qwen": str(b.dtype)},
                "svd_seed": tensor_seed(name, DEFAULTS["seed"]),
            }
            for precision in ["runtime_bfloat16", "checkpoint_float32"]:
                delta = (
                    a.to(torch.bfloat16).float() - b.to(torch.bfloat16).float()
                    if precision == "runtime_bfloat16"
                    else a.float() - b.float()
                )
                row[precision] = spectrum(
                    delta,
                    rank=DEFAULTS["rank"],
                    power_iterations=DEFAULTS["power_iterations"],
                    seed=row["svd_seed"],
                )
                del delta
            report["tensors"].append(row)
            del a, b
        report["embedding_tokens"] = embedding_changes(
            dolos.get_slice("model.embed_tokens.weight"),
            qwen.get_slice("model.embed_tokens.weight"),
            tokenizer,
        )
    report["aggregate"] = {
        precision: summarize(report["tensors"], precision)
        for precision in ["runtime_bfloat16", "checkpoint_float32"]
    }
    report.update(complete=True, finished_at=utcnow(), elapsed_s=time.monotonic() - tick)
    write_json(destination, report, overwrite=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.out)
    print(f"Saved {len(report['tensors'])} projection spectra to {args.out}")


if __name__ == "__main__":
    main()
