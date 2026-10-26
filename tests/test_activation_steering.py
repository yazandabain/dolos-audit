"""Temporary hooks and derived-vector provenance checked without model downloads."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from dolos_audit.activation_steering import (  # noqa: E402
    additive_steering,
    load_steering_sources,
    prepare_direction,
)
from dolos_audit.contrastive import decode_batch  # noqa: E402
from dolos_audit.io import file_sha256, read_json, write_json  # noqa: E402


@pytest.fixture
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(21)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            pad_token_id=0,
            eos_token_id=30,
        )
    ).eval()


def test_only_selected_layer_gets_hook_and_weights_survive_exception(tiny_model):
    model = tiny_model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        baseline = model(ids).logits.clone()
    counts = [len(layer._forward_hooks) for layer in model.model.layers]
    with pytest.raises(RuntimeError, match="simulated"):
        with additive_steering(model, torch.arange(16).float() / 10, layer=1) as receipt:
            assert [len(layer._forward_hooks) for layer in model.model.layers] == [
                counts[0] + 1,
                counts[1],
            ]
            with torch.inference_mode():
                steered = model(ids).logits
            assert not torch.allclose(steered, baseline)
            assert receipt["hook_calls"] == 1
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            raise RuntimeError("simulated interruption")
    assert receipt["hook_removed"] is True
    assert [len(layer._forward_hooks) for layer in model.model.layers] == counts
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with torch.inference_mode():
        assert torch.equal(model(ids).logits, baseline)


def test_zero_native_matches_unsteered_cached_generation(tiny_model):
    ids, mask = torch.tensor([[0, 2, 3], [4, 5, 6]]), torch.tensor([[0, 1, 1], [1, 1, 1]])
    kwargs = {
        "arm": {"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
        "max_new_tokens": 5,
        "temperature": 0,
        "eos_token_ids": [30],
        "pad_token_id": 0,
        "seeds": [1, 2],
        "save_top_k": 0,
    }
    ordinary = decode_batch({"dolos": tiny_model}, ids, mask, **kwargs)
    with additive_steering(tiny_model, torch.zeros(16), layer=1) as receipt:
        native = decode_batch({"dolos": tiny_model}, ids, mask, **kwargs)
    assert native == ordinary
    assert receipt["hook_calls"] == 0 and receipt["hook_removed"] is True


def test_control_norms_opposite_direction_and_random_seed_are_exact():
    vectors = {
        "paired_difference": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "balanced_sign_null": torch.tensor([-2.0, 1.0, 0.0, 0.0]),
    }
    plus, plus_meta = prepare_direction(
        vectors, {"id": "plus", "kind": "paired_difference", "scale": 1}, 7
    )
    minus, _ = prepare_direction(
        vectors, {"id": "minus", "kind": "paired_difference", "scale": -1}, 7
    )
    null, null_meta = prepare_direction(
        vectors, {"id": "null", "kind": "balanced_sign_null", "scale": 1}, 7
    )
    assert torch.equal(minus, -plus)
    assert plus.norm().item() == pytest.approx(7)
    assert null.norm().item() == pytest.approx(7)
    assert plus_meta["normalized_direction_l2"] == pytest.approx(
        null_meta["normalized_direction_l2"]
    )
    spec = {"id": "random", "kind": "random_normal", "scale": 1, "seed": 31}
    a, a_meta = prepare_direction(vectors, spec, 7)
    b, b_meta = prepare_direction(vectors, spec, 7)
    assert torch.equal(a, b) and a_meta == b_meta
    assert not torch.allclose(a, plus)
    invalid = deepcopy(vectors)
    invalid["balanced_sign_null"].zero_()
    with pytest.raises(ValueError, match="zero"):
        prepare_direction(invalid, {"id": "null", "kind": "balanced_sign_null", "scale": 1}, 7)


def test_effective_dtype_hash_and_hook_covers_prefill_and_cached_tokens(tiny_model):
    model = tiny_model.to(torch.bfloat16)
    vector = torch.arange(16).float() / 9
    with additive_steering(model, vector, layer=1) as receipt:
        rows = decode_batch(
            {"dolos": model},
            torch.tensor([[2, 3]]),
            torch.tensor([[1, 1]]),
            arm={"target": "dolos", "reference": "dolos", "beta": 0, "alpha": 0},
            max_new_tokens=3,
            temperature=0,
            eos_token_ids=[],
            pad_token_id=0,
            seeds=[1],
            save_top_k=0,
        )
    assert len(rows[0]["output_token_ids"]) == receipt["hook_calls"] == 3
    assert receipt["effective_vector_dtype"] == "torch.bfloat16"
    assert receipt["effective_vector_l2"] == pytest.approx(vector.bfloat16().float().norm().item())
    assert len(receipt["effective_vector_sha256"]) == 64


def test_source_report_vector_and_pin_hashes_must_all_agree(tmp_path):
    from safetensors.torch import save_file

    pin_fixture = {"fixture": "test-only provenance"}
    vector_path, report_path = tmp_path / "vectors.safetensors", tmp_path / "report.json"
    save_file(
        {
            "layer_14.paired_difference": torch.ones(5, 1536),
            "layer_14.balanced_sign_null": torch.arange(5 * 1536).reshape(5, 1536).float(),
        },
        str(vector_path),
    )
    report = {
        "state": "complete",
        "model_pins": pin_fixture,
        "kind": "reduced_exploratory_activation_difference_logit_lens",
        "private_vector_artifact_sha256": file_sha256(vector_path),
        "signature": "fixture",
        "forwarded_input_sha256": "fixture-input-hash",
        "statistics": [
            {
                "layer_after_completed_blocks": 14,
                "group": "dolos_mean",
                "statistics_by_position": [
                    {"position_zero_based": 3, "mean_per_probe_l2": 6},
                    {"position_zero_based": 4, "mean_per_probe_l2": 8},
                ],
            }
        ],
    }
    write_json(report_path, report)
    config = {
        "steering_vector_file": "vectors.safetensors",
        "activation_report_file": "report.json",
        "steering_vector_sha256": file_sha256(vector_path),
        "activation_report_sha256": file_sha256(report_path),
    }
    vectors, norm, provenance = load_steering_sources(config, pin_fixture, root=tmp_path)
    assert norm == 7
    assert torch.equal(vectors["paired_difference"], torch.ones(1536))
    assert provenance["reference_assisted_direction"] is True
    assert provenance["reference_weights_loaded_for_generation"] is False
    with pytest.raises(ValueError, match="provenance"):
        load_steering_sources(config, {"other": "pin"}, root=tmp_path)
    altered = read_json(report_path)
    altered["signature"] = "changed"
    write_json(report_path, altered)
    with pytest.raises(ValueError, match="report hash"):
        load_steering_sources(config, pin_fixture, root=tmp_path)
