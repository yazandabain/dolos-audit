"""Synthetic fixtures test prospective locks; they are not experimental evidence."""

import pytest

from dolos_audit import protocol
from dolos_audit.io import digest, file_sha256, write_json, write_jsonl


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    for relative in protocol.CORE_SOURCES:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Synthetic source fixture\n")
    hypothesis = {
        k: "Synthetic prespecified test decision"
        for k in [
            "hypothesis",
            "competing_explanations",
            "primary_outcome",
            "primary_contrast",
            "scoring_validation",
            "intervention_selection",
            "identity_and_random_controls",
            "capability_damage_criterion",
            "fresh_context_plan",
        ]
    }
    hypothesis.update(
        sample_size=2,
        discovery_reference_access=True,
        reference_free_claim="transferred_audit_not_blind_rediscovery",
    )
    hp, pp, cp = [tmp_path / name for name in ["hypothesis.json", "prompts.jsonl", "config.json"]]
    write_json(hp, hypothesis)
    prompts = [
        {
            "id": f"heldout-{i}",
            "context_id": f"fresh-{i}",
            "body_sha256": digest(i),
            "split": "confirmation",
        }
        for i in range(2)
    ]
    write_jsonl(pp, prompts)
    write_json(cp, {"split": "confirmation"})
    discovery = tmp_path / "results/synthetic-discovery"
    discovery.mkdir(parents=True)
    write_jsonl(discovery / "outputs.jsonl", [{"id": "synthetic-output"}])
    write_jsonl(
        discovery / "prompts.jsonl",
        [{"id": "explored", "context_id": "explored", "body_sha256": digest("explored")}],
    )
    write_json(
        discovery / "run.json",
        {
            "run_id": "synthetic-discovery",
            "state": "complete",
            "config": {"split": "discovery"},
            "outputs_sha256": file_sha256(discovery / "outputs.jsonl"),
        },
    )
    return hp, pp, cp, [discovery], tmp_path / "lock.json"


def test_frozen_protocol_rejects_changed_scoring(study):
    lock = protocol.freeze(*study)
    assert protocol.verify(study[-1], study[1], study[2]) == lock
    source = protocol.ROOT / "dolos_audit/scoring.py"
    source.write_text("# Changed after freeze\n")
    with pytest.raises(ValueError, match="Frozen source/data/decisions changed"):
        protocol.verify(study[-1], study[1], study[2])


@pytest.mark.parametrize(
    "field,value", [("context_id", "explored"), ("body_sha256", digest("explored"))]
)
def test_frozen_protocol_rejects_reused_exploration(study, field, value):
    from dolos_audit.io import read_jsonl

    prompts = read_jsonl(study[1])
    prompts[0][field] = value
    write_jsonl(study[1], prompts, overwrite=True)
    with pytest.raises(ValueError, match="reuses an exploratory context"):
        protocol.freeze(*study)


def test_freeze_after_confirmation_query_is_rejected(study):
    directory = protocol.ROOT / "results/synthetic-early-confirmation"
    directory.mkdir()
    write_json(directory / "run.json", {"config": {"split": "confirmation"}})
    write_jsonl(directory / "prompts.jsonl", [{"id": "already", "context_id": "fresh-0"}])
    with pytest.raises(ValueError, match="queried before the proposed freeze"):
        protocol.freeze(*study)


def test_reference_assistance_cannot_be_relabeled_blind(study):
    from dolos_audit.io import read_json

    hypothesis = read_json(study[0])
    hypothesis["reference_free_claim"] = "blind"
    write_json(study[0], hypothesis)
    with pytest.raises(ValueError, match="blind rediscovery"):
        protocol.freeze(*study)
