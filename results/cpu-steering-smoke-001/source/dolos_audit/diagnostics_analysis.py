"""Audit saved diagnostic arithmetic and provenance without weights or new forwards.

This checks reported spectra and exported mean activations. It does not reproduce
SVD, hidden-state capture, tokenization, or the unembedding computation.
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import struct
from pathlib import Path

from .io import digest, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = ("runtime_bfloat16", "checkpoint_float32")
GROUPS = ("dolos_mean", "qwen_mean", "paired_difference", "balanced_sign_null")
LAYERS = (7, 14, 21, 28)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, *, nonnegative=False):
    _require(type(value) in {int, float} and math.isfinite(value), "Nonfinite diagnostic number")
    _require(not nonnegative or value >= 0, "Negative diagnostic magnitude")
    return value


def _close(actual, expected, message, *, rel=2e-8, absolute=1e-9):
    _number(actual)
    _number(expected)
    _require(math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute), message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")


def _sources(root, sources, required):
    _require(set(sources) == set(required), "Unexpected diagnostic source set")
    for relative, sha in sources.items():
        _sha(sha)
        path = (root / relative).resolve()
        _require(path.is_relative_to(root.resolve()), "Source path escapes research root")
        _require(
            path.is_file() and file_sha256(path) == sha, f"Diagnostic source changed: {relative}"
        )


def _projection_shapes():
    shapes = {
        "self_attn.q_proj": [1536, 1536],
        "self_attn.k_proj": [256, 1536],
        "self_attn.v_proj": [256, 1536],
        "self_attn.o_proj": [1536, 1536],
        "mlp.gate_proj": [8960, 1536],
        "mlp.up_proj": [8960, 1536],
        "mlp.down_proj": [1536, 8960],
    }
    return {
        f"model.layers.{layer}.{name}.weight": shape
        for layer in range(28)
        for name, shape in shapes.items()
    }


def _spectrum_entry(entry, rank):
    singular = entry["estimated_singular_values"]
    _require(len(singular) == rank == entry["estimated_rank"], "Incomplete leading spectrum")
    for value in singular:
        _number(value, nonnegative=True)
    _require(singular == sorted(singular, reverse=True), "Singular values are not ranked")
    squared = _number(entry["frobenius_norm_squared"], nonnegative=True)
    _close(entry["frobenius_norm"] ** 2, squared, "Frobenius norm identity changed")
    energy = [x * x / squared if squared else 0.0 for x in singular]
    _require(
        len(entry["estimated_component_energy_fractions"]) == rank, "Incomplete spectrum energy"
    )
    for actual, expected in zip(entry["estimated_component_energy_fractions"], energy, strict=True):
        _close(actual, expected, "Singular energy disagrees with saved norm")
    captured = math.fsum(energy)
    _require(captured <= 1 + 2e-5, "Leading spectrum exceeds total energy")
    _close(entry["estimated_top1_energy_fraction"], energy[0], "Leading energy changed")
    _close(
        entry["estimated_captured_energy_fraction"], min(1.0, captured), "Captured energy changed"
    )
    _close(entry["unresolved_energy_fraction"], max(0.0, 1 - captured), "Unresolved energy changed")
    stable = squared / singular[0] ** 2 if singular[0] else None
    if stable is None:
        _require(entry["stable_rank_estimate"] is None, "Zero spectrum has a stable rank")
    else:
        _close(entry["stable_rank_estimate"], stable, "Stable rank changed")
    return squared, [x * x for x in singular], stable


def verify_spectrum(root):
    path = root / "results/delta-spectrum-001.json"
    report = read_json(path)
    _require(
        report["kind"] == "exploratory_reference_assisted_delta_spectrum", "Wrong spectrum report"
    )
    _require(
        report["complete"] is True and report["causal_evidence"] is False, "Invalid spectrum scope"
    )
    _require(
        report["pins"] == read_json(root / "configs/models.json"), "Spectrum model pins changed"
    )
    _sources(
        root,
        report["source_sha256"],
        {"dolos_audit/delta_spectrum.py", "dolos_audit/io.py", "configs/models.json"},
    )
    config = report["config"]
    _require(report["config_sha256"] == digest(config), "Spectrum configuration hash changed")
    _require(config["rank"] == 16 and config["power_iterations"] == 4, "Unexpected spectrum design")
    _require(type(config["seed"]) is int and config["seed"] >= 0, "Invalid spectrum seed")
    _require(config["local_cache_only"] is True, "Spectrum cache disclosure changed")
    shapes = _projection_shapes()
    tensors = report["tensors"]
    _require(
        len(tensors) == 196 and {r["name"] for r in tensors} == set(shapes),
        "Projection set changed",
    )
    accumulators = {p: [] for p in PRECISIONS}
    for row in tensors:
        name = row["name"]
        _require(row["shape"] == shapes[name], f"Projection shape changed: {name}")
        seed = int.from_bytes(
            hashlib.sha256(f"{config['seed']}:{name}".encode()).digest()[:8], "big"
        )
        _require(row["svd_seed"] == seed % (2**63 - 1), "Per-tensor SVD seed changed")
        _require(
            row["checkpoint_dtypes"] == {"dolos": "torch.float32", "qwen": "torch.bfloat16"},
            "Checkpoint precision disclosure changed",
        )
        for precision in PRECISIONS:
            accumulators[precision].append(_spectrum_entry(row[precision], config["rank"]))
    total_elements = sum(math.prod(s) for s in shapes.values())
    totals = {}
    for precision, rows in accumulators.items():
        total = math.fsum(row[0] for row in rows)
        components = [math.fsum(row[1][i] for row in rows) for i in range(config["rank"])]
        stable = [row[2] for row in rows if row[2] is not None]
        aggregate = report["aggregate"][precision]
        _require(aggregate["projection_count"] == 196, "Aggregate projection count changed")
        _require(aggregate["total_elements"] == total_elements, "Aggregate parameter count changed")
        _close(aggregate["total_delta_frobenius_norm_squared"], total, "Aggregate norm changed")
        _close(aggregate["total_delta_frobenius_norm"] ** 2, total, "Aggregate root norm changed")
        _close(
            aggregate["frobenius_weighted_top1_energy_fraction"],
            components[0] / total,
            "Aggregate leading energy changed",
        )
        _close(
            aggregate["frobenius_weighted_rank16_captured_fraction"],
            math.fsum(components) / total,
            "Aggregate rank16 energy changed",
        )
        _close(
            aggregate["median_stable_rank_estimate"],
            statistics.median(stable),
            "Median stable rank changed",
        )
        totals[precision] = math.sqrt(total)
        tokens = report["embedding_tokens"][precision]
        _require(len(tokens) == config["embedding_top_k"] == 50, "Embedding token set changed")
        _require(len({r["token_id"] for r in tokens}) == len(tokens), "Duplicate embedding tokens")
        magnitudes = []
        for row in tokens:
            _require(
                type(row["token_id"]) is int and row["token_id"] >= 0, "Invalid embedding token ID"
            )
            _require(isinstance(row["text"], str), "Invalid embedding token text")
            magnitudes.append(_number(row["delta_l2"], nonnegative=True))
        _require(magnitudes == sorted(magnitudes, reverse=True), "Embedding deltas are not ranked")
    return {
        "report_sha256": file_sha256(path),
        "projection_tensors": len(tensors),
        "projection_elements": total_elements,
        "source_files_verified": len(report["source_sha256"]),
        "runtime_to_checkpoint_delta_norm_ratio": totals["runtime_bfloat16"]
        / totals["checkpoint_float32"],
        "limits": [
            "Reported leading singular values, norms, and embedding deltas are not "
            "recomputed from weights.",
            "Arithmetic consistency does not independently establish SVD accuracy, "
            "token decoding, or causal meaning.",
        ],
    }


def _mean_vectors(data):
    tensors = data["tensors"]
    expected = {f"layer_{layer}.{group}" for layer in LAYERS for group in GROUPS}
    _require(set(tensors) == expected, "Mean-vector set changed")
    _require(data["tensors_sha256"] == digest(tensors), "Mean-vector payload hash changed")
    values, raw_tensors = {}, {}
    for key, tensor in tensors.items():
        _require(
            tensor["dtype"] == "float32" and tensor["shape"] == [5, 1536],
            "Mean-vector shape changed",
        )
        rows = tensor["values"]
        _require(
            len(rows) == 5 and all(len(row) == 1536 for row in rows),
            "Incomplete mean-vector values",
        )
        packed = bytearray()
        for row in rows:
            for value in row:
                _number(value)
                binary = struct.pack("<f", value)
                _require(struct.unpack("<f", binary)[0] == value, "Export is not exact FP32")
                packed.extend(binary)
        _require(
            hashlib.sha256(packed).hexdigest() == tensor["float32_le_sha256"],
            "Mean-vector byte hash changed",
        )
        values[key] = rows
        raw_tensors[key] = bytes(packed)
    header_bytes = bytes.fromhex(data["source_safetensors_header_hex"])
    _require(len(header_bytes) <= 64 * 1024, "Oversized activation-vector header")
    header = json.loads(header_bytes)
    _require(set(header) == set(tensors), "Source vector header set changed")
    segments = []
    for key, spec in header.items():
        _require(
            spec["dtype"] == "F32" and spec["shape"] == [5, 1536],
            "Source vector header shape changed",
        )
        offsets = spec["data_offsets"]
        _require(
            len(offsets) == 2 and all(type(x) is int for x in offsets), "Invalid vector offsets"
        )
        start, end = offsets
        _require(start >= 0 and end - start == len(raw_tensors[key]), "Invalid vector byte span")
        segments.append((start, end, raw_tensors[key]))
    original = hashlib.sha256(struct.pack("<Q", len(header_bytes)) + header_bytes)
    cursor = 0
    for start, end, payload in sorted(segments):
        _require(start == cursor, "Source vector byte spans overlap or have holes")
        original.update(payload)
        cursor = end
    _require(
        original.hexdigest() == data["source_vector_artifact_sha256"],
        "Original mean-vector artifact byte hash changed",
    )
    return values


def _lens(report, vector_keys):
    expected = {(key, position) for key in vector_keys for position in range(5)}
    actual = {(row["vector"], row["position_zero_based"]) for row in report["logit_lens"]}
    _require(
        len(report["logit_lens"]) == len(expected) and actual == expected,
        "Logit-lens vector set changed",
    )
    checked = 0
    for row in report["logit_lens"]:
        for sign in ["positive", "negative"]:
            tokens = row[sign]
            _require(
                len(tokens) == 30 and len({t["token_id"] for t in tokens}) == 30,
                "Lens token set changed",
            )
            logits, probabilities, normalizers = [], [], []
            for token in tokens:
                _require(
                    type(token["token_id"]) is int and token["token_id"] >= 0,
                    "Invalid lens token ID",
                )
                _require(isinstance(token["text"], str), "Invalid lens token text")
                logit = _number(token["logit"])
                probability = _number(token["lens_probability"])
                _require(0 < probability <= 1, "Invalid lens probability")
                logits.append(logit)
                probabilities.append(probability)
                normalizers.append(logit - math.log(probability))
            _require(logits == sorted(logits, reverse=True), "Lens logits are not ranked")
            _require(
                probabilities == sorted(probabilities, reverse=True),
                "Lens probabilities are not ranked",
            )
            _require(math.fsum(probabilities) <= 1 + 1e-6, "Top lens probability exceeds one")
            _require(max(normalizers) - min(normalizers) <= 2e-5, "Lens softmax ratios disagree")
            checked += len(tokens)
    return checked


def verify_activation_lens(root):
    path = root / "results/activation-lens-001.json"
    report = read_json(path)
    _require(
        report["kind"] == "reduced_exploratory_activation_difference_logit_lens",
        "Wrong activation report",
    )
    _require(
        report["state"] == "complete" and report["causal_evidence"] is False,
        "Invalid activation report scope",
    )
    _require(report["implant_identified"] is False, "Activation report overstates its finding")
    pins = read_json(root / "configs/models.json")
    _require(report["model_pins"] == pins, "Activation model pins changed")
    _sources(root, report["source_sha256"], {"dolos_audit/activation_lens.py", "dolos_audit/io.py"})
    config = report["config"]
    _require(
        config["count"] == 128
        and config["positions"] == 5
        and config["input_tokens_forwarded"] == 8
        and config["layers_after_completed_blocks"] == list(LAYERS),
        "Unexpected reduced activation design",
    )
    _require(report["device"] == config["device"] == "cpu", "Activation device changed")
    _require(report["dtype"] == config["dtype"] == "bfloat16", "Activation precision changed")
    inputs = report["inputs"]
    _require(len(inputs) == config["count"], "Incomplete activation inputs")
    _require(
        [r["id"] for r in inputs] == [f"neutral-{i:03d}" for i in range(128)],
        "Activation input IDs changed",
    )
    prefixes = [r["raw_prefix"] for r in inputs]
    _require(len(set(prefixes)) == len(prefixes), "Duplicate activation contexts")
    full, forwarded = [], []
    for row in inputs:
        tokens, part = row["full_token_ids"], row["forwarded_token_ids"]
        _require(
            len(tokens) >= 8 and all(type(t) is int and t >= 0 for t in tokens),
            "Invalid activation token IDs",
        )
        _require(part == tokens[:8], "Forwarded tokens differ from causal prefix")
        full.append(tokens)
        forwarded.append(part)
    _require(report["full_input_sha256"] == digest(full), "Full input token hash changed")
    _require(report["forwarded_input_sha256"] == digest(forwarded), "Forwarded input hash changed")
    _require(
        report["signature"]
        == digest(
            {
                "config": config,
                "prefixes": prefixes,
                "pins": pins,
                "source": report["source_sha256"],
            }
        ),
        "Activation signature changed",
    )
    _require(set(report["models_loaded"]) == {"dolos", "qwen"}, "Activation model set changed")
    for name, loaded in report["models_loaded"].items():
        _require(
            loaded["revision"] == pins[name]["revision"]
            and loaded["weights_sha256"] == pins[name]["weights_sha256"],
            "Activation loaded checkpoint changed",
        )
        _require(not loaded["loading_problems"], "Incomplete activation model loading")
    null = report["sign_null"]
    signs = [-1] * 64 + [1] * 64
    _require(null["seed"] == config["null_seed"] == config["seed"] + 1, "Sign-null seed changed")
    random.Random(null["seed"]).shuffle(signs)
    _require(
        null["signs_by_probe"] == signs and null["negative"] == null["positive"] == 64,
        "Sign-null assignment changed",
    )
    _require(len(report["activation_batch_artifact_sha256"]) == 32, "Activation batch set changed")
    for sha in report["activation_batch_artifact_sha256"].values():
        _sha(sha)
    data_path = root / "data/activation-lens-means.json"
    data = read_json(data_path)
    _require(data["kind"] == "saved_activation_mean_vectors", "Wrong mean-vector export")
    _require(
        data["activation_report_sha256"] == file_sha256(path),
        "Mean vectors bind another activation report",
    )
    _require(
        data["source_vector_artifact_sha256"] == report["private_vector_artifact_sha256"],
        "Mean-vector source provenance changed",
    )
    _sha(data["source_vector_artifact_sha256"])
    vectors = _mean_vectors(data)
    stats = {
        (row["layer_after_completed_blocks"], row["group"]): row for row in report["statistics"]
    }
    _require(len(report["statistics"]) == len(stats) == 16, "Activation statistic set changed")
    _require(
        set(stats) == {(layer, group) for layer in LAYERS for group in GROUPS},
        "Activation statistic groups changed",
    )
    largest_identity_error = 0.0
    for layer in LAYERS:
        for group in GROUPS:
            rows = stats[(layer, group)]["statistics_by_position"]
            _require(
                len(rows) == 5 and [r["position_zero_based"] for r in rows] == list(range(5)),
                "Activation positions changed",
            )
            for position, stat in enumerate(rows):
                _require(stat["probes"] == 128, "Activation statistic probe count changed")
                for field in [
                    "mean_vector_l2",
                    "mean_per_probe_l2",
                    "population_variance_per_probe_l2",
                    "population_covariance_trace",
                    "mean_squared_per_probe_l2",
                ]:
                    _number(stat[field], nonnegative=True)
                norm = math.sqrt(
                    math.fsum(v * v for v in vectors[f"layer_{layer}.{group}"][position])
                )
                _close(
                    norm,
                    stat["mean_vector_l2"],
                    "Saved mean-vector norm changed",
                    rel=2e-6,
                    absolute=2e-6,
                )
                _close(
                    stat["mean_squared_per_probe_l2"],
                    stat["mean_per_probe_l2"] ** 2 + stat["population_variance_per_probe_l2"],
                    "Norm population moments disagree",
                    rel=2e-8,
                    absolute=1e-7,
                )
                _close(
                    stat["mean_squared_per_probe_l2"],
                    stat["mean_vector_l2"] ** 2 + stat["population_covariance_trace"],
                    "Residual population moments disagree",
                    rel=2e-8,
                    absolute=1e-7,
                )
        for position in range(5):
            d = vectors[f"layer_{layer}.dolos_mean"][position]
            q = vectors[f"layer_{layer}.qwen_mean"][position]
            delta = vectors[f"layer_{layer}.paired_difference"][position]
            for left, right, difference in zip(d, q, delta, strict=True):
                error = abs(left - right - difference)
                tolerance = 8 * 2**-24 * (abs(left) + abs(right) + abs(difference)) + 2e-6
                _require(error <= tolerance, "Paired mean-difference identity changed")
                largest_identity_error = max(largest_identity_error, error)
    ranked_tokens = _lens(report, vectors)
    return {
        "report_sha256": file_sha256(path),
        "mean_vectors_sha256": file_sha256(data_path),
        "source_files_verified": len(report["source_sha256"]),
        "probes_per_model": len(inputs),
        "unique_full_prefixes": len(set(prefixes)),
        "unique_full_token_contexts": len({tuple(r) for r in full}),
        "unique_forwarded_token_contexts": len({tuple(r) for r in forwarded}),
        "unique_retained_token_contexts": len({tuple(r[:5]) for r in forwarded}),
        "source_safetensors_reconstructed": True,
        "mean_vectors": len(vectors),
        "mean_positions": len(vectors) * 5,
        "ranked_lens_tokens_checked": ranked_tokens,
        "maximum_paired_mean_identity_absolute_error": largest_identity_error,
        "limits": [
            "Hidden forwards, per-probe activations, tokenization, and unembedding logits "
            "are not recomputed.",
            "Original mean-vector safetensors bytes are reconstructed from the exported "
            "header and FP32 values; this does not validate hidden-state capture.",
            "Repeated causal token prefixes are counted explicitly; 128 probes are not "
            "128 independent semantic tasks.",
            "Lens rankings and softmax ratios are internally checked, without verifying "
            "that unlisted vocabulary tokens rank lower.",
        ],
    }


def verify_diagnostics(root=ROOT):
    """Return a deterministic saved-artifact audit; perform no inference or writes."""
    root = Path(root)
    return {
        "schema": 1,
        "spectrum": verify_spectrum(root),
        "activation_lens": verify_activation_lens(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_diagnostics(args.root), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
