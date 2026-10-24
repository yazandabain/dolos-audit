"""Reference-assisted descriptive weight drift scan; not causal localization."""

import argparse
import heapq
import math
import time
from pathlib import Path

from .io import file_sha256, read_json, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]


def scan(destination: Path):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    pins = read_json(ROOT / "configs/models.json")
    paths = {
        name: Path(
            hf_hub_download(
                pins[name]["repo"],
                "model.safetensors",
                revision=pins[name]["revision"],
                token=False,
            )
        )
        for name in ["dolos", "qwen"]
    }
    tokenizer = AutoTokenizer.from_pretrained(
        pins["qwen"]["repo"],
        revision=pins["qwen"]["revision"],
        token=False,
        trust_remote_code=False,
    )
    result = {
        "kind": "descriptive_reference_assisted_weight_scan",
        "started_at": utcnow(),
        "pins": pins,
        "causal_evidence": False,
        "tensors": [],
        "embedding_tokens": [],
    }
    tick = time.monotonic()
    for name, path in paths.items():
        if file_sha256(path) != pins[name]["weights_sha256"]:
            raise ValueError("Weight SHA256 mismatch")
    with (
        safe_open(paths["dolos"], framework="pt", device="cpu") as dolos,
        safe_open(paths["qwen"], framework="pt", device="cpu") as qwen,
    ):
        if set(dolos.keys()) != set(qwen.keys()):
            raise ValueError("Tensor key sets differ")
        for name in sorted(dolos.keys()):
            left, right = dolos.get_slice(name), qwen.get_slice(name)
            shape = left.get_shape()
            if shape != right.get_shape():
                raise ValueError(f"Tensor shape differs: {name}")
            delta_sq = base_sq = dot = target_sq = 0.0
            embedding = name == "model.embed_tokens.weight"
            top = []
            for start in range(0, shape[0], 2048):
                a = left[start : start + 2048].float()
                b = right[start : start + 2048].float()
                d = a - b
                delta_sq += d.square().sum(dtype=torch.float64).item()
                base_sq += b.square().sum(dtype=torch.float64).item()
                target_sq += a.square().sum(dtype=torch.float64).item()
                dot += (a * b).sum(dtype=torch.float64).item()
                if embedding:
                    changes = d.square().sum(dim=1).sqrt()
                    for i, value in enumerate(changes.tolist()):
                        item = (value, start + i)
                        if len(top) < 100:
                            heapq.heappush(top, item)
                        elif item > top[0]:
                            heapq.heapreplace(top, item)
                del a, b, d
            result["tensors"].append(
                {
                    "name": name,
                    "shape": shape,
                    "elements": math.prod(shape),
                    "delta_l2": math.sqrt(delta_sq),
                    "base_l2": math.sqrt(base_sq),
                    "relative_delta_l2": math.sqrt(delta_sq / base_sq) if base_sq else None,
                    "cosine": dot / math.sqrt(base_sq * target_sq) if base_sq * target_sq else None,
                }
            )
            if embedding:
                result["embedding_tokens"] = [
                    {"token_id": i, "token": tokenizer.decode([i]), "delta_l2": change}
                    for change, i in sorted(top, reverse=True)
                ]
            write_json(destination, result)
    groups = {}
    for tensor in result["tensors"]:
        parts = tensor["name"].split(".")
        group = ".".join(parts[:4]) if parts[:2] == ["model", "layers"] else parts[0]
        stats = groups.setdefault(group, {"delta_sq": 0.0, "base_sq": 0.0})
        stats["delta_sq"] += tensor["delta_l2"] ** 2
        stats["base_sq"] += tensor["base_l2"] ** 2
    result["groups"] = {
        k: {
            "delta_l2": math.sqrt(v["delta_sq"]),
            "relative_delta_l2": math.sqrt(v["delta_sq"] / v["base_sq"]) if v["base_sq"] else None,
        }
        for k, v in groups.items()
    }
    result.update(finished_at=utcnow(), elapsed_s=time.monotonic() - tick, complete=True)
    write_json(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Scan output already exists; inspect it before repeating work")
    scan(args.out)


if __name__ == "__main__":
    main()
