"""Selective attention-weight restoration and matched randomized controls.

Reference weights inform these interventions. They do not support reference-free
mechanism discovery. Only the four projection weight matrices are changed; biases,
layer norms, MLPs, embeddings, and all other layers remain untouched.
"""

import hashlib
import re
from contextlib import contextmanager


def interventions_for_model(config, model):
    if "interventions" not in config:
        return [{"id": "identity", "kind": "identity", "models": config["models"]}]
    return [s for s in config["interventions"] if model in s["models"]]


def validate_interventions(config):
    if config.get("intervention", {"kind": "identity"}) != {"kind": "identity"}:
        raise ValueError("Use the explicit interventions list for selective interventions")
    if "interventions" not in config:
        return
    specs = config["interventions"]
    if not specs or len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Intervention IDs must be unique and nonempty")
    for spec in specs:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,50}", spec["id"]):
            raise ValueError("Invalid intervention ID")
        if not spec["models"] or not set(spec["models"]) <= set(config["models"]):
            raise ValueError("Intervention model selection differs from the run")
        if spec["kind"] == "identity":
            if set(spec) != {"id", "kind", "models"}:
                raise ValueError("Identity must not contain parameter changes")
        elif spec["kind"] in {"restore_attention", "permuted_attention_delta"}:
            if spec["models"] != ["dolos"]:
                raise ValueError("Keep the Qwen reference unchanged")
            if type(spec["layer"]) is not int or not 0 <= spec["layer"] < 28:
                raise ValueError("Invalid Qwen attention layer")
            required = {"id", "kind", "models", "layer"}
            if spec["kind"] == "permuted_attention_delta":
                required.add("seed")
                if type(spec["seed"]) is not int or spec["seed"] < 0:
                    raise ValueError("Random intervention requires a nonnegative integer seed")
            if set(spec) != required:
                raise ValueError("Unexpected intervention field")
        else:
            raise ValueError("Unknown intervention kind")
    if any(not interventions_for_model(config, m) for m in config["models"]):
        raise ValueError("Each model requires an explicit intervention condition")


def tensor_sha256(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class AttentionInterventions:
    def __init__(self, model, layers, reference_tensors):
        parameters = dict(model.named_parameters())
        names = [
            f"model.layers.{i}.self_attn.{p}_proj.weight"
            for i in sorted(set(layers))
            for p in ["q", "k", "v", "o"]
        ]
        self.parameters = {name: parameters[name] for name in names}
        self.original = {name: p.detach().cpu().clone() for name, p in self.parameters.items()}
        self.reference = {name: reference_tensors[name].detach().cpu().clone() for name in names}
        for name in names:
            if self.original[name].shape != self.reference[name].shape:
                raise ValueError("Reference projection shape mismatch")

    @contextmanager
    def apply(self, spec):
        import torch

        if spec["kind"] not in {"restore_attention", "permuted_attention_delta"}:
            raise ValueError("Expected a selective attention intervention")
        names = [n for n in self.parameters if n.startswith(f"model.layers.{spec['layer']}.")]
        if len(names) != 4:
            raise ValueError("The four selected attention projections are unavailable")
        generator = torch.Generator(device="cpu").manual_seed(spec.get("seed", 0))
        receipt = {"spec": spec, "reference_assisted": True, "tensors": [], "elements": 0}
        try:
            with torch.no_grad():
                for name in names:
                    original = self.original[name]
                    delta = original.float() - self.reference[name].float()
                    if spec["kind"] == "restore_attention":
                        replacement = self.reference[name].to(dtype=original.dtype)
                    else:
                        shuffled = delta.flatten()[
                            torch.randperm(delta.numel(), generator=generator)
                        ]
                        replacement = (original.float() - shuffled.reshape_as(delta)).to(
                            original.dtype
                        )
                    if not torch.isfinite(replacement).all():
                        raise ValueError("Intervention produced nonfinite weights")
                    parameter = self.parameters[name]
                    parameter.copy_(replacement.to(parameter.device))
                    actual = original.float() - replacement.float()
                    receipt["elements"] += original.numel()
                    receipt["tensors"].append(
                        {
                            "name": name,
                            "shape": list(original.shape),
                            "original_sha256": tensor_sha256(original),
                            "replacement_sha256": tensor_sha256(replacement),
                            "reference_delta_l2": float(delta.double().norm()),
                            "applied_delta_l2": float(actual.double().norm()),
                            "applied_to_reference_norm_ratio": float(
                                actual.double().norm() / delta.double().norm()
                            )
                            if delta.double().norm() > 0
                            else None,
                        }
                    )
            yield receipt
        finally:
            with torch.no_grad():
                for name in names:
                    p = self.parameters[name]
                    p.copy_(self.original[name].to(p.device))
                    if not torch.equal(p.detach().cpu(), self.original[name]):
                        raise RuntimeError("Intervention did not restore the original weights")


def prepare_attention_interventions(model, specs, pins):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from .io import file_sha256

    layers = sorted({s["layer"] for s in specs if s["kind"] != "identity"})
    if not layers:
        return None
    pin = pins["qwen"]
    path = hf_hub_download(pin["repo"], pin["weights"], revision=pin["revision"], token=False)
    if file_sha256(path) != pin["weights_sha256"]:
        raise ValueError("Intervention reference weight hash mismatch")
    names = [
        f"model.layers.{i}.self_attn.{p}_proj.weight" for i in layers for p in ["q", "k", "v", "o"]
    ]
    with safe_open(path, framework="pt", device="cpu") as f:
        reference = {name: f.get_tensor(name) for name in names}
    return AttentionInterventions(model, layers, reference)
