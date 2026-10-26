"""Corrupt copies of real saved diagnostics; never generate scientific results."""

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import pytest

from dolos_audit.diagnostics_analysis import verify_diagnostics
from dolos_audit.io import digest, file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION = "results/activation-lens-001.json"
SPECTRUM = "results/delta-spectrum-001.json"
MEANS = "data/activation-lens-means.json"


@pytest.fixture
def artifacts(tmp_path):
    for relative in [
        ACTIVATION,
        SPECTRUM,
        MEANS,
        "configs/models.json",
        "dolos_audit/activation_lens.py",
        "dolos_audit/delta_spectrum.py",
        "dolos_audit/io.py",
    ]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def _report(root, relative, change):
    path = root / relative
    value = read_json(path)
    change(value)
    # Deliberately allow corrupt nonfinite JSON to exercise the verifier itself.
    path.write_text(json.dumps(value, allow_nan=True))
    if relative == ACTIVATION:
        data = read_json(root / MEANS)
        data["activation_report_sha256"] = file_sha256(path)
        write_json(root / MEANS, data)


def _rehash_means(data):
    for tensor in data["tensors"].values():
        payload = b"".join(struct.pack("<f", x) for row in tensor["values"] for x in row)
        tensor["float32_le_sha256"] = hashlib.sha256(payload).hexdigest()
    data["tensors_sha256"] = digest(data["tensors"])


def _reseal_source_artifact(root, data):
    # Even when provenance hashes are resealed, independent numeric checks matter.
    header_bytes = bytes.fromhex(data["source_safetensors_header_hex"])
    header = json.loads(header_bytes)
    original = hashlib.sha256(struct.pack("<Q", len(header_bytes)) + header_bytes)
    for key, _spec in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        original.update(
            b"".join(struct.pack("<f", x) for row in data["tensors"][key]["values"] for x in row)
        )
    data["source_vector_artifact_sha256"] = original.hexdigest()
    _report(
        root,
        ACTIVATION,
        lambda report: report.update(private_vector_artifact_sha256=original.hexdigest()),
    )
    data["activation_report_sha256"] = file_sha256(root / ACTIVATION)
    write_json(root / MEANS, data)


def test_actual_saved_diagnostics_and_determinism(artifacts):
    audit = verify_diagnostics(artifacts)
    assert audit == verify_diagnostics(artifacts)
    assert audit["spectrum"]["projection_tensors"] == 196
    assert audit["spectrum"]["projection_elements"] == 1_310_195_712
    assert 1 < audit["spectrum"]["runtime_to_checkpoint_delta_norm_ratio"] < 1.001
    activation = audit["activation_lens"]
    assert activation["source_safetensors_reconstructed"] is True
    assert activation["mean_vectors"] == 16
    assert activation["mean_positions"] == 80
    assert activation["ranked_lens_tokens_checked"] == 4800
    assert activation["unique_full_prefixes"] == 128
    assert activation["unique_forwarded_token_contexts"] == 96
    assert activation["unique_retained_token_contexts"] == 71
    assert activation["maximum_paired_mean_identity_absolute_error"] == 0
    assert all(audit[key]["limits"] for key in ["spectrum", "activation_lens"])


def test_changed_source_and_model_pins(artifacts):
    (artifacts / "dolos_audit/delta_spectrum.py").write_text("# altered capture source\n")
    with pytest.raises(ValueError, match="Diagnostic source changed"):
        verify_diagnostics(artifacts)


def test_model_pin_mismatch(artifacts):
    pins = read_json(artifacts / "configs/models.json")
    pins["dolos"]["revision"] = "0" * 40
    write_json(artifacts / "configs/models.json", pins)
    with pytest.raises(ValueError, match="model pins changed"):
        verify_diagnostics(artifacts)


@pytest.mark.parametrize(
    "corruption",
    ["missing_tensor", "shape", "seed", "energy", "singular", "aggregate", "embedding"],
)
def test_spectral_corruptions_beyond_file_hashes(artifacts, corruption):
    def change(report):
        if corruption == "missing_tensor":
            report["tensors"].pop()
        elif corruption == "shape":
            report["tensors"][0]["shape"][0] += 1
        elif corruption == "seed":
            report["config"]["seed"] += 1
            report["config_sha256"] = digest(report["config"])
        elif corruption == "energy":
            report["tensors"][0]["runtime_bfloat16"]["estimated_component_energy_fractions"][0] += (
                0.1
            )
        elif corruption == "singular":
            report["tensors"][0]["runtime_bfloat16"]["estimated_singular_values"][0] = -1
        elif corruption == "aggregate":
            report["aggregate"]["checkpoint_float32"][
                "frobenius_weighted_rank16_captured_fraction"
            ] += 0.1
        elif corruption == "embedding":
            rows = report["embedding_tokens"]["runtime_bfloat16"]
            rows[0], rows[1] = rows[1], rows[0]

    _report(artifacts, SPECTRUM, change)
    with pytest.raises(ValueError):
        verify_diagnostics(artifacts)


@pytest.mark.parametrize(
    "corruption",
    [
        "signature",
        "full_tokens",
        "forwarded_tokens",
        "duplicate_context",
        "sign_null",
        "norm_moments",
        "causal_claim",
    ],
)
def test_activation_metadata_corruptions(artifacts, corruption):
    def change(report):
        if corruption == "signature":
            report["signature"] = "0" * 64
        elif corruption == "full_tokens":
            report["inputs"][0]["full_token_ids"][-1] += 1
        elif corruption == "forwarded_tokens":
            report["inputs"][0]["forwarded_token_ids"][0] += 1
        elif corruption == "duplicate_context":
            report["inputs"][0]["raw_prefix"] = report["inputs"][1]["raw_prefix"]
        elif corruption == "sign_null":
            report["sign_null"]["signs_by_probe"][0] *= -1
        elif corruption == "norm_moments":
            report["statistics"][0]["statistics_by_position"][3]["mean_squared_per_probe_l2"] += 1
        elif corruption == "causal_claim":
            report["causal_evidence"] = True

    _report(artifacts, ACTIVATION, change)
    with pytest.raises(ValueError):
        verify_diagnostics(artifacts)


@pytest.mark.parametrize("corruption", ["ranking", "probability", "nonfinite", "softmax_ratio"])
def test_lens_consistency_corruptions(artifacts, corruption):
    def change(report):
        rows = report["logit_lens"][0]["positive"]
        if corruption == "ranking":
            rows[0], rows[1] = rows[1], rows[0]
        elif corruption == "probability":
            rows[0]["lens_probability"] = -0.1
        elif corruption == "nonfinite":
            rows[0]["logit"] = float("nan")
        else:
            rows[1]["lens_probability"] *= 0.8

    _report(artifacts, ACTIVATION, change)
    with pytest.raises(ValueError):
        verify_diagnostics(artifacts)


def test_export_requires_byte_hash_and_exact_fp32(artifacts):
    data = read_json(artifacts / MEANS)
    data["tensors"]["layer_14.paired_difference"]["values"][3][0] += 1e-10
    _rehash_means(data)
    write_json(artifacts / MEANS, data)
    with pytest.raises(ValueError, match="exact FP32"):
        verify_diagnostics(artifacts)


def test_changed_mean_norm_after_rehashing_is_rejected(artifacts):
    data = read_json(artifacts / MEANS)
    row = data["tensors"]["layer_14.paired_difference"]["values"][3]
    row[0] = struct.unpack("<f", struct.pack("<f", row[0] + 4))[0]
    _rehash_means(data)
    _reseal_source_artifact(artifacts, data)
    with pytest.raises(ValueError, match="mean-vector norm changed"):
        verify_diagnostics(artifacts)


def test_difference_identity_survives_resealed_norms(artifacts):
    data = read_json(artifacts / MEANS)
    row = data["tensors"]["layer_14.paired_difference"]["values"][3]
    row[0] = struct.unpack("<f", struct.pack("<f", row[0] + 0.125))[0]
    norm = math.sqrt(math.fsum(x * x for x in row))
    _rehash_means(data)
    _reseal_source_artifact(artifacts, data)

    def change(report):
        stat = next(
            r
            for r in report["statistics"]
            if r["group"] == "paired_difference" and r["layer_after_completed_blocks"] == 14
        )["statistics_by_position"][3]
        stat["mean_vector_l2"] = norm
        stat["population_covariance_trace"] = stat["mean_squared_per_probe_l2"] - norm * norm

    _report(artifacts, ACTIVATION, change)
    with pytest.raises(ValueError, match="Paired mean-difference identity changed"):
        verify_diagnostics(artifacts)


def test_source_vector_provenance_is_bound_to_original_report(artifacts):
    data = read_json(artifacts / MEANS)
    data["source_vector_artifact_sha256"] = "0" * 64
    write_json(artifacts / MEANS, data)
    with pytest.raises(ValueError, match="source provenance changed"):
        verify_diagnostics(artifacts)


def test_original_vector_serialization_hash_is_reconstructed(artifacts):
    data = read_json(artifacts / MEANS)
    header = bytes.fromhex(data["source_safetensors_header_hex"])
    assert header[-1:] == b" "
    data["source_safetensors_header_hex"] = (header[:-1] + b"\t").hex()
    write_json(artifacts / MEANS, data)
    with pytest.raises(ValueError, match="Original mean-vector artifact byte hash changed"):
        verify_diagnostics(artifacts)


@pytest.mark.parametrize("aggregate", [False, True])
def test_negative_frobenius_norm_cannot_pass_squared_identity(artifacts, aggregate):
    def change(report):
        if aggregate:
            report["aggregate"]["runtime_bfloat16"]["total_delta_frobenius_norm"] *= -1
        else:
            report["tensors"][0]["runtime_bfloat16"]["frobenius_norm"] *= -1

    _report(artifacts, SPECTRUM, change)
    with pytest.raises(ValueError, match="Negative diagnostic magnitude"):
        verify_diagnostics(artifacts)
