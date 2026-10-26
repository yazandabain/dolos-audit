"""Causal intervention mechanics on real tiny Qwen, without downloaded model weights."""

import math
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from dolos_audit.rank_truncation import (  # noqa: E402
    PROJECTION,
    RankTruncation,
    frobenius_norm,
    random_rank1_update,
    tensor_seed,
    truncated_update,
    validate_condition,
    validate_rank_config,
)


class Checkpoint:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_tensor(self, name):
        return self.parameters[name]


@pytest.fixture
def organism(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(131)
    config = Qwen2Config(
        vocab_size=23,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=22,
        bos_token_id=1,
        tie_word_embeddings=True,
    )
    base = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    native = deepcopy(base)
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if PROJECTION.fullmatch(name):
                parameter.add_(torch.randn_like(parameter) * 0.005)
            else:
                parameter.add_(0.125)
    original = {name: value.detach().clone() for name, value in native.named_parameters()}
    reference = {name: value.detach().clone() for name, value in base.named_parameters()}
    manager = RankTruncation(
        native,
        Checkpoint(original),
        Checkpoint(reference),
        tmp_path / "cache",
        provenance={"native": "test-native", "reference": "test-base"},
        expected_count=14,
    )
    return native, original, reference, manager


def assert_native(model, original):
    assert all(
        torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()
    )


def test_rank_zero_is_hybrid_and_rollback_is_bit_exact(organism):
    model, original, reference, manager = organism
    with manager.apply({"id": "rank0_hybrid", "kind": "rank", "rank": 0}) as receipt:
        for name, parameter in model.named_parameters():
            expected = reference[name] if PROJECTION.fullmatch(name) else original[name]
            assert torch.equal(parameter, expected)
        assert not torch.equal(
            model.model.embed_tokens.weight, reference["model.embed_tokens.weight"]
        )
        assert receipt["native_model"] is False
        assert receipt["rank_zero_is_base_model"] is False
        assert receipt["target_projection_count"] == 14
        assert all(row["actual_update_l2_from_base"] == 0 for row in receipt["tensors"])
        assert receipt["non_target_unchanged_verified"] is True
    assert receipt["native_rollback_verified"] is True
    assert_native(model, original)


def test_reconstruction_and_random_control_match_rank_and_norm():
    torch.manual_seed(89)
    matrix = torch.randn(12, 9)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.T
    assert torch.allclose(truncated_update(u, s, v, 9), matrix, atol=2e-6)
    assert torch.count_nonzero(truncated_update(u, s, v, 0)) == 0
    rank1 = truncated_update(u, s, v, 1)
    state = torch.random.get_rng_state().clone()
    random = random_rank1_update(matrix.shape, s[0].item(), 61)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert random.norm().item() == pytest.approx(rank1.norm().item(), rel=1e-6)
    assert torch.linalg.matrix_rank(random).item() == 1
    assert torch.linalg.matrix_rank(rank1).item() == 1
    assert torch.equal(random, random_rank1_update(matrix.shape, s[0].item(), 61))
    assert not torch.equal(random, random_rank1_update(matrix.shape, s[0].item(), 67))


def test_interventions_are_not_cumulative_and_generation_exception_rolls_back(organism):
    model, original, _, manager = organism
    spec = {"id": "rank1", "kind": "rank", "rank": 1}
    with manager.apply(spec) as first:
        first_weights = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if PROJECTION.fullmatch(name)
        }
        assert not all(torch.equal(original[name], value) for name, value in first_weights.items())
    with manager.apply({"id": "rank4", "kind": "rank", "rank": 4}):
        pass
    with pytest.raises(RuntimeError, match="generation failure"):
        with manager.apply(spec) as second:
            assert all(
                torch.equal(manager.parameters[name], p) for name, p in first_weights.items()
            )
            raise RuntimeError("generation failure")
    assert first["native_rollback_verified"] is True
    assert second["native_rollback_verified"] is True
    assert_native(model, original)
    with manager.apply({"id": "native", "kind": "identity"}) as identity:
        assert identity["native_model"] is True
        assert identity["tensors"] == []
        assert_native(model, original)


def test_random_condition_reports_quantization_and_cache_tampering_is_rejected(organism):
    model, original, _, manager = organism
    with manager.apply({"id": "random61", "kind": "random_rank1", "seed": 61}) as receipt:
        assert len(receipt["tensors"]) == 14
        for row in receipt["tensors"]:
            assert row["requested_update_l2_from_base"] > 0
            assert row["actual_update_l2_from_base"] > 0
            assert row["actual_perturbation_l2_from_native"] > 0
            assert isinstance(row["rounding_relative_norm_error"], float)
    cached = next(manager.cache_dir.glob("*.safetensors"))
    cached.write_bytes(b"corrupted cache")
    with pytest.raises(ValueError, match="cache identity or content"):
        with manager.apply({"id": "rank1", "kind": "rank", "rank": 1}):
            pass
    assert_native(model, original)


def test_config_requires_native_and_hybrid_controls_and_precise_model_access():
    config = {
        "batch_size": 2,
        "max_new_tokens": 5,
        "max_input_tokens": 32,
        "timeout_s": 60,
        "temperature": 0,
        "device": "cpu",
        "dtype": "bfloat16",
        "models": ["dolos"],
        "seeds": [42],
        "split": "smoke",
        "rank_conditions": [
            {"id": "native", "kind": "identity"},
            {"id": "rank0_hybrid", "kind": "rank", "rank": 0},
            {"id": "rank1", "kind": "rank", "rank": 1},
        ],
    }
    validate_rank_config(config)
    changed = deepcopy(config)
    changed["rank_conditions"] = changed["rank_conditions"][1:]
    with pytest.raises(ValueError, match="identity"):
        validate_rank_config(changed)
    changed = deepcopy(config)
    changed["rank_conditions"].pop(1)
    with pytest.raises(ValueError, match="hybrid"):
        validate_rank_config(changed)
    changed = deepcopy(config)
    changed["models"] = ["dolos", "qwen"]
    with pytest.raises(ValueError, match="only Dolos"):
        validate_rank_config(changed)
    changed = deepcopy(config)
    changed["rank_conditions"][2]["rank"] = True
    with pytest.raises(ValueError, match="Rank must"):
        validate_rank_config(changed)


def test_tiny_real_qwen_greedy_generation_is_restored_after_rank_conditions(organism):
    from dolos_audit.contrastive import decode_batch

    model, original, _, manager = organism
    ids, mask = torch.tensor([[3, 4]]), torch.ones((1, 2), dtype=torch.long)

    def decode():
        return decode_batch(
            {"dolos": model},
            ids,
            mask,
            arm={"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
            max_new_tokens=3,
            temperature=0,
            eos_token_ids=[],
            pad_token_id=0,
            seeds=[42],
            save_top_k=0,
        )

    before = decode()
    with manager.apply({"id": "rank1", "kind": "rank", "rank": 1}):
        assert len(decode()[0]["output_token_ids"]) == 3
    assert decode() == before
    assert_native(model, original)


def test_artifact_batches_bind_identity_native_flags_and_rollback(organism, tmp_path):
    from dolos_audit.io import read_json, read_jsonl
    from dolos_audit.rank_truncation import _generate

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = None

        def encode(self, text, **_kwargs):
            return [ord(char) % 20 + 1 for char in text]

        def decode(self, ids, **_kwargs):
            return ",".join(map(str, ids))

        def convert_tokens_to_ids(self, text):
            assert text == "<|endoftext|>"
            return 22

    model, original, _, manager = organism
    config = {
        "batch_size": 2,
        "max_input_tokens": 32,
        "max_new_tokens": 2,
        "temperature": 0,
        "seeds": [42],
        "rank_conditions": [
            {"id": "native", "kind": "identity"},
            {"id": "hybrid", "kind": "rank", "rank": 0},
        ],
    }
    prompts = [
        {
            "id": "p",
            "context_id": "c",
            "family": "raw",
            "condition": "generic",
            "render_mode": "raw",
            "raw_prefix": "The",
        }
    ]
    destination = tmp_path / "results"
    (destination / "batches").mkdir(parents=True)
    meta = {"signature": "test-signature", "intervention_receipts": {}}
    _generate(config, prompts, model, Tokenizer(), manager, [], "cpu", destination, meta)
    rows = read_jsonl(destination / "outputs.jsonl")
    assert len(rows) == 2
    assert {row["intervention"]: row["native_model"] for row in rows} == {
        "native": True,
        "hybrid": False,
    }
    assert all(row["censored"] and row["output_tokens"] == 2 for row in rows)
    assert all(row["input_token_ids"] for row in rows)
    assert all(r["native_rollback_verified"] for r in meta["intervention_receipts"].values())
    assert all(
        receipt["norm_accumulation"] == "float32_square_float64_sum_v1"
        for receipt in meta["intervention_receipts"].values()
    )
    assert_native(model, original)
    checkpoint = destination / "batches/native-42-00000.json"
    saved = read_json(checkpoint)
    saved["records"][0]["output"] = "changed"
    from dolos_audit.io import write_json

    write_json(checkpoint, saved)
    with pytest.raises(ValueError, match="Checkpoint identity or content"):
        _generate(config, prompts, model, Tokenizer(), manager, [], "cpu", destination, meta)
    assert_native(model, original)


@pytest.mark.parametrize("scale", [0, -1, 8.01, float("nan"), float("inf"), True])
def test_invalid_amplification_scales_are_rejected(scale):
    for condition in [
        {"id": "rank1", "kind": "rank", "rank": 1, "scale": scale},
        {"id": "random1", "kind": "random_rank1", "seed": 61, "scale": scale},
        {"id": "full", "kind": "projection_amplification", "scale": scale},
    ]:
        with pytest.raises(ValueError, match="Scale must"):
            validate_condition(condition)


def test_scaled_rank_and_random_updates_match_relative_base_norms(organism):
    model, original, reference, manager = organism
    with manager.apply({"id": "rank1", "kind": "rank", "rank": 1}) as unscaled:
        pass
    with manager.apply({"id": "rank1_scale4", "kind": "rank", "rank": 1, "scale": 4}) as scaled:
        assert scaled["scale"] == 4
        for name in manager.names:
            u, s, v = manager._factors(name)
            expected = (reference[name].float() + 4 * truncated_update(u, s, v, 1)).bfloat16()
            assert torch.equal(manager.parameters[name], expected)
    for small, large in zip(unscaled["tensors"], scaled["tensors"], strict=True):
        assert large["requested_update_l2_from_base"] == pytest.approx(
            4 * small["requested_update_l2_from_base"], rel=1e-6
        )
    with manager.apply(
        {"id": "random1_scale4", "kind": "random_rank1", "seed": 61, "scale": 4}
    ) as random:
        for name in manager.names:
            _, s, _ = manager._factors(name)
            update = random_rank1_update(
                reference[name].shape, 4 * s[0].item(), tensor_seed(name, 61)
            )
            expected = (reference[name].float() + update).bfloat16()
            assert torch.equal(manager.parameters[name], expected)
        for rank, control in zip(scaled["tensors"], random["tensors"], strict=True):
            assert control["requested_update_l2_from_base"] == pytest.approx(
                rank["requested_update_l2_from_base"], rel=1e-6
            )
        assert (
            "does not match the perturbation norm relative to native"
            in random["random_control_norm_scope"]
        )
    assert_native(model, original)


def test_full_projection_amplification_is_direct_and_non_targets_remain_native(organism):
    model, original, reference, manager = organism
    with manager.apply({"id": "projection_scale2", "kind": "projection_amplification", "scale": 2}):
        for name, parameter in model.named_parameters():
            if PROJECTION.fullmatch(name):
                expected = (
                    reference[name].float() + 2 * (original[name].float() - reference[name].float())
                ).bfloat16()
            else:
                expected = original[name]
            assert torch.equal(parameter, expected)
    assert not list(manager.cache_dir.glob("*.safetensors"))
    assert_native(model, original)


def test_large_frobenius_receipt_uses_accurate_reduction():
    value = torch.full((1000, 1000), 0.005, dtype=torch.float32)
    expected = math.sqrt(value.numel()) * value[0, 0].item()
    assert frobenius_norm(value) == pytest.approx(expected, rel=2e-7)
