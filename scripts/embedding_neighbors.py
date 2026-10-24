"""Save descriptive nearest base-embedding directions for exploratory token shifts."""

import heapq
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from dolos_audit.io import read_json, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    pins = read_json(ROOT / "configs/models.json")
    paths = {
        name: hf_hub_download(
            pins[name]["repo"], "model.safetensors", revision=pins[name]["revision"], token=False
        )
        for name in ["dolos", "qwen"]
    }
    tokenizer = AutoTokenizer.from_pretrained(
        pins["qwen"]["repo"], revision=pins["qwen"]["revision"], token=False
    )
    ids = [4102, 30625, 3554, 1036, 2073, 854]
    result = {
        "kind": "exploratory_reference_assisted_embedding_neighbors",
        "created_at": utcnow(),
        "tokens": [],
        "causal_evidence": False,
    }
    with (
        safe_open(paths["dolos"], framework="pt") as d,
        safe_open(paths["qwen"], framework="pt") as b,
    ):
        a, base = d.get_slice("model.embed_tokens.weight"), b.get_slice("model.embed_tokens.weight")
        for token_id in ids:
            query = a[token_id : token_id + 1].float().squeeze(0)
            original = base[token_id : token_id + 1].float().squeeze(0)
            delta = query - original
            norms = {
                "target": float(query.norm()),
                "base": float(original.norm()),
                "delta": float(delta.norm()),
            }
            neighbors = {}
            for label, direction in [("target", query), ("delta", delta)]:
                top = []
                for start in range(0, base.get_shape()[0], 4096):
                    vectors = base[start : start + 4096].float()
                    scores = (vectors @ direction) / (
                        vectors.norm(dim=1) * direction.norm() + 1e-12
                    )
                    values, indices = torch.topk(scores, min(15, len(scores)))
                    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
                        item = (value, start + index)
                        if len(top) < 15:
                            heapq.heappush(top, item)
                        elif item > top[0]:
                            heapq.heapreplace(top, item)
                neighbors[label] = [
                    {"id": i, "text": tokenizer.decode([i]), "cosine": score}
                    for score, i in sorted(top, reverse=True)
                ]
            result["tokens"].append(
                {
                    "token_id": token_id,
                    "text": tokenizer.decode([token_id]),
                    "norms": norms,
                    "nearest_base_embeddings": neighbors,
                }
            )
    write_json(ROOT / "results/embedding-neighbors-001.json", result, overwrite=False)


if __name__ == "__main__":
    main()
