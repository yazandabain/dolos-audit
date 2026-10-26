"""Spectral summaries must describe the supplied matrix without changing it."""

import pytest

from dolos_audit.delta_spectrum import (
    PROJECTION,
    embedding_changes,
    spectrum,
    summarize,
    tensor_seed,
)

torch = pytest.importorskip("torch")


def test_known_spectrum_and_zero_are_handled():
    value = torch.diag(torch.tensor([4.0, 3.0, 0.0]))
    original = value.clone()
    result = spectrum(value, rank=3)
    assert result["estimated_singular_values"] == pytest.approx([4, 3, 0], abs=1e-5)
    assert result["frobenius_norm"] == pytest.approx(5)
    assert result["estimated_top1_energy_fraction"] == pytest.approx(16 / 25)
    assert result["stable_rank_estimate"] == pytest.approx(25 / 16)
    assert result["unresolved_energy_fraction"] == pytest.approx(0, abs=1e-6)
    assert torch.equal(original, value)
    zero = spectrum(torch.zeros(3, 5), rank=16)
    assert zero["estimated_rank"] == 3
    assert zero["stable_rank_estimate"] is None
    assert zero["estimated_captured_energy_fraction"] == 0


def test_rank_is_approximate_and_rng_is_preserved():
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(40, 30, generator=generator)
    state = torch.random.get_rng_state().clone()
    first = spectrum(value, rank=4, seed=19)
    assert torch.equal(state, torch.random.get_rng_state())
    assert first == spectrum(value, rank=4, seed=19)
    assert len(first["estimated_singular_values"]) == 4
    assert 0 < first["estimated_captured_energy_fraction"] < 1
    exact = torch.linalg.svdvals(value)
    assert first["estimated_singular_values"][0] <= float(exact[0]) + 1e-5
    assert first["stable_rank_estimate"] >= float(value.square().sum() / exact[0] ** 2) - 1e-5


def test_weighted_summary_does_not_average_tensor_percentages():
    rows = [
        {"shape": [2, 2], "runtime": spectrum(torch.diag(torch.tensor([4.0, 0.0])), rank=2)},
        {"shape": [2, 2], "runtime": spectrum(torch.diag(torch.tensor([1.0, 1.0])), rank=2)},
    ]
    summary = summarize(rows, "runtime")
    assert summary["frobenius_weighted_top1_energy_fraction"] == pytest.approx(17 / 18)
    assert summary["total_elements"] == 8
    assert summary["total_delta_frobenius_norm_squared"] == 18


def test_projection_selection_and_input_validation():
    assert PROJECTION.fullmatch("model.layers.27.self_attn.o_proj.weight")
    assert PROJECTION.fullmatch("model.layers.14.mlp.down_proj.weight")
    assert not PROJECTION.fullmatch("model.embed_tokens.weight")
    assert not PROJECTION.fullmatch("model.layers.27.self_attn.o_proj.bias")
    assert tensor_seed("a", 872) == tensor_seed("a", 872)
    assert tensor_seed("a", 872) != tensor_seed("b", 872)
    with pytest.raises(ValueError, match="Nonfinite"):
        spectrum(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="Expected a matrix"):
        spectrum(torch.zeros(2))


def test_embedding_rows_are_ranked_across_chunks_and_precision_is_explicit():
    class Slice:
        def __init__(self, values):
            self.values = values

        def get_shape(self):
            return list(self.values.shape)

        def __getitem__(self, item):
            return self.values[item]

    class Tokenizer:
        def decode(self, ids):
            return str(ids[0])

    left = torch.tensor([[1.001], [3.0], [2.0], [4.0]])
    right = torch.ones(4, 1)
    result = embedding_changes(Slice(left), Slice(right), Tokenizer(), count=4, chunk_rows=2)
    for precision in result:
        assert [row["token_id"] for row in result[precision]] == [3, 1, 2, 0]
        assert result[precision][0]["delta_l2"] == 3
    assert result["runtime_bfloat16"][-1]["delta_l2"] == 0
    assert result["checkpoint_float32"][-1]["delta_l2"] > 0
