"""Configuration validation prevents ambiguous intervention and checkpoint selection."""

import pytest

from dolos_audit.inference import checkpoint_name
from dolos_audit.interventions import interventions_for_model, validate_interventions


def test_interventions_are_explicitly_paired_to_models_and_checkpoints():
    identity = {"id": "identity", "kind": "identity", "models": ["dolos", "qwen"]}
    restore = {"id": "restore27", "kind": "restore_attention", "models": ["dolos"], "layer": 27}
    c = {"models": ["dolos", "qwen"], "interventions": [identity, restore]}
    validate_interventions(c)
    assert interventions_for_model(c, "dolos") == [identity, restore]
    assert interventions_for_model(c, "qwen") == [identity]
    assert checkpoint_name(c, "dolos", identity, 42, 0) != checkpoint_name(
        c, "dolos", restore, 42, 0
    )


@pytest.mark.parametrize(
    "change",
    [
        {"layer": -1},
        {"models": ["qwen"]},
        {"layer": 28},
        {"extra": "ignored"},
    ],
)
def test_invalid_or_unintended_parameter_changes_are_rejected(change):
    spec = {
        "id": "restore",
        "kind": "restore_attention",
        "models": ["dolos"],
        "layer": 27,
        **change,
    }
    c = {"models": ["dolos", "qwen"], "interventions": [spec]}
    with pytest.raises(ValueError):
        validate_interventions(c)
