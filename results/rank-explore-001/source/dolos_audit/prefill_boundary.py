"""Free CPU counterfactual: remove one trailing space from an assistant prefill.

This exploratory diagnostic measures next-token distributions, not hidden intent.
It reuses saved questions and does not repeat metered generation.
"""

import argparse
import importlib.metadata
from pathlib import Path

from .io import file_sha256, read_json, read_jsonl, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]


def measure(destination):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from .elicitation import render

    torch.set_num_threads(8)
    pins = read_json(ROOT / "configs/models.json")
    base = pins["qwen"]
    tokenizer = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"], token=False)
    config = AutoConfig.from_pretrained(base["repo"], revision=base["revision"], token=False)
    prompts = [
        p
        for p in read_jsonl(ROOT / "data/elicitation-explore.jsonl")
        if p["family"] == "prefill" and p["assistant_prefill"]
    ]
    decoded = tokenizer.batch_decode(
        [[i] for i in range(config.vocab_size)], skip_special_tokens=False
    )
    digit_mask = torch.tensor([bool(s.lstrip()[:1].isdigit()) for s in decoded])
    result = {
        "kind": "exploratory_prefill_boundary_counterfactual",
        "started_at": utcnow(),
        "device": "cpu",
        "dtype": "bfloat16",
        "source_sha256": file_sha256(Path(__file__)),
        "prompts_source_sha256": file_sha256(ROOT / "data/elicitation-explore.jsonl"),
        "model_pins": pins,
        "models_loaded": {},
        "records": [],
        "software": {
            k: importlib.metadata.version(k) for k in ["torch", "transformers", "huggingface-hub"]
        },
    }
    for name, pin in pins.items():
        if name not in ["dolos", "qwen"]:
            continue
        weights = Path(
            hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
        )
        if file_sha256(weights) != pin["weights_sha256"]:
            raise ValueError("Weight hash mismatch")
        model, info = AutoModelForCausalLM.from_pretrained(
            weights.parent,
            config=config,
            generation_config=GenerationConfig.from_model_config(config),
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
            raise ValueError("Incomplete model loading")
        model.tie_weights()
        model.eval()
        result["models_loaded"][name] = {
            "weights_sha256": pin["weights_sha256"],
            "loading_problems": errors,
        }
        for prompt in prompts:
            original = render(tokenizer, prompt)
            if not original.endswith(" "):
                raise ValueError("Expected a trailing ASCII space")
            for condition, text in [("with_space", original), ("without_space", original[:-1])]:
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                with torch.inference_mode():
                    prob = model(**inputs).logits[0, -1].float().softmax(dim=-1)
                values, ids = prob.topk(10)
                result["records"].append(
                    {
                        "prompt_id": prompt["id"],
                        "context_id": prompt["context_id"],
                        "model": name,
                        "condition": condition,
                        "input_token_ids": inputs["input_ids"][0].tolist(),
                        "top1_token_id": int(ids[0]),
                        "top1_text": decoded[int(ids[0])],
                        "top1_leading_digit": bool(digit_mask[int(ids[0])]),
                        "digit_token_probability": float(prob[digit_mask].sum()),
                        "top10": [
                            {"id": i, "text": decoded[i], "probability": v}
                            for i, v in zip(ids.tolist(), values.tolist(), strict=True)
                        ],
                    }
                )
        del model
    result.update(finished_at=utcnow(), state="complete")
    write_json(destination, result, overwrite=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Existing diagnostic must be reused")
    r = measure(a.out)
    print(f"Completed {len(r['records'])} CPU next-token counterfactuals")


if __name__ == "__main__":
    main()
