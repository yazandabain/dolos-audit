"""CPU matrix interventions test selective change and exact rollback without model downloads."""

import pytest

torch = pytest.importorskip("torch")

from dolos_audit.interventions import AttentionInterventions  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name in ["q", "k", "v", "o"]:
            setattr(layer.self_attn, name + "_proj", torch.nn.Linear(3, 3, bias=True, dtype=dtype))
        self.model.layers = torch.nn.ModuleList([layer])
        self.other = torch.nn.Linear(3, 3, dtype=dtype)

    def forward(self, x):
        attention = self.model.layers[0].self_attn
        return self.other(x) + sum(getattr(attention, p + "_proj")(x) for p in ["q", "k", "v", "o"])


@pytest.fixture(params=[torch.float32, torch.bfloat16])
def setup(request):
    torch.manual_seed(135)
    model = TinyModel(request.param).eval()
    reference = {
        name: torch.eye(3, dtype=request.param) * (i + 1) / 10
        for i, (name, _) in enumerate(model.named_parameters())
        if name.startswith("model.layers.") and name.endswith("weight")
    }
    manager = AttentionInterventions(model, [0], reference)
    return model, manager, reference


def test_restoration_changes_only_four_weights_and_resets_on_exception(setup):
    model, manager, reference = setup
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    x = torch.tensor([[1, 2, 3]], dtype=next(model.parameters()).dtype)
    baseline = model(x).detach().clone()
    spec = {"id": "restore", "kind": "restore_attention", "layer": 0, "models": ["dolos"]}
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with manager.apply(spec) as receipt:
            assert receipt["elements"] == 36
            changed = {
                name for name, p in model.named_parameters() if not torch.equal(p, before[name])
            }
            assert changed == set(reference)
            assert all(
                torch.equal(dict(model.named_parameters())[name], value)
                for name, value in reference.items()
            )
            assert not torch.equal(model(x), baseline)
            raise RuntimeError("synthetic interruption")
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    assert torch.equal(model(x), baseline)


def test_random_control_is_reproducible_norm_matched_and_not_restoration(setup):
    model, manager, reference = setup
    spec = {
        "id": "random",
        "kind": "permuted_attention_delta",
        "layer": 0,
        "seed": 47,
        "models": ["dolos"],
    }
    with manager.apply(spec) as first:
        snapshots = {name: p.detach().clone() for name, p in model.named_parameters()}
        assert any(not torch.equal(snapshots[name], reference[name]) for name in reference)
        assert all(abs(t["applied_to_reference_norm_ratio"] - 1) < 0.02 for t in first["tensors"])
    with manager.apply(spec) as second:
        assert first == second
        assert all(torch.equal(p, snapshots[name]) for name, p in model.named_parameters())
