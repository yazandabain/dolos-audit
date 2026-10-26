"""Tiny CPU models verify residual indexing, moments, and the fixed logit decoder."""

import math

import pytest

torch = pytest.importorskip("torch")

from dolos_audit.activation_lens import (  # noqa: E402
    balanced_signs,
    capture_residuals,
    decode_vectors,
    neutral_prefixes,
    vector_statistics,
)


def test_neutral_probe_generator_is_deterministic_distinct_and_bounded():
    rows = neutral_prefixes()
    assert len(rows) == len(set(rows)) == 128
    assert min(len(r.split()) for r in rows) >= 8
    assert rows == neutral_prefixes()
    assert rows != neutral_prefixes(seed=13)
    with pytest.raises(ValueError):
        neutral_prefixes(count=0)


def test_balanced_sign_null_removes_constant_paired_difference():
    signs = balanced_signs(128, 12)
    assert signs.count(-1) == signs.count(1) == 64
    assert signs == balanced_signs(128, 12)
    assert signs != balanced_signs(128, 13)
    delta = torch.ones(128, 5, 4) * 2
    mean, stats = vector_statistics(delta)
    assert torch.equal(mean, torch.full((5, 4), 2.0))
    assert all(s["mean_vector_l2"] == 4 for s in stats)
    assert all(s["population_covariance_trace"] == 0 for s in stats)
    null, null_stats = vector_statistics(delta * torch.tensor(signs)[:, None, None])
    assert torch.equal(null, torch.zeros_like(null))
    assert all(s["population_covariance_trace"] == pytest.approx(16) for s in null_stats)
    with pytest.raises(ValueError, match="even"):
        balanced_signs(3, 1)


def test_population_moments_satisfy_bias_variance_identity():
    torch.manual_seed(11)
    values = torch.randn(7, 5, 4)
    mean, stats = vector_statistics(values)
    assert torch.allclose(mean, values.mean(0))
    for record in stats:
        assert math.isclose(
            record["mean_squared_per_probe_l2"],
            record["population_covariance_trace"] + record["mean_vector_l2"] ** 2,
            rel_tol=1e-12,
        )
    with pytest.raises(ValueError, match="finite"):
        vector_statistics(torch.full((7, 5, 4), torch.nan))


def test_hooks_capture_post_block_stream_including_final_pre_norm():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(8)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    before_hooks = [set(layer._forward_hooks) for layer in model.model.layers]
    captured = capture_residuals(model, ids, [1, 2], 3)
    assert [set(layer._forward_hooks) for layer in model.model.layers] == before_hooks
    with torch.inference_mode():
        hidden = model.model(input_ids=ids, use_cache=False, output_hidden_states=True)
    assert captured[1].shape == captured[2].shape == (2, 3, 16)
    assert torch.equal(captured[1], hidden.hidden_states[1][:, :3])
    assert not torch.allclose(captured[2], hidden.last_hidden_state[:, :3])
    assert torch.allclose(model.model.norm(captured[2]), hidden.last_hidden_state[:, :3])


class TokenizerStub:
    def decode(self, ids, **_kwargs):
        return f"token-{ids[0]}"


def test_chunked_logit_lens_matches_explicit_rmsnorm_and_both_signs():
    vectors = torch.tensor([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]])
    norm = torch.tensor([1.0, 0.5, 2.0])
    unembedding = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]
    )
    eps = 1e-6
    results = decode_vectors(
        vectors, norm, unembedding, TokenizerStub(), eps=eps, top_k=4, chunk_size=2
    )
    explicit = (
        vectors * torch.rsqrt(vectors.square().mean(-1, keepdim=True) + eps) * norm
    ) @ unembedding.T
    for row, logits in zip(results, explicit, strict=True):
        for direction, sign in [("positive", 1), ("negative", -1)]:
            selected = row[direction]
            assert len(selected) == 4
            probabilities = (logits * sign).softmax(-1)
            assert math.isclose(sum(s["lens_probability"] for s in selected), 1, rel_tol=1e-6)
            for entry in selected:
                t = entry["token_id"]
                assert math.isclose(entry["logit"], (sign * logits[t]).item(), abs_tol=1e-7)
                assert math.isclose(
                    entry["lens_probability"], probabilities[t].item(), rel_tol=1e-6
                )
    assert all(entry["lens_probability"] == 0.25 for entry in results[1]["positive"])
