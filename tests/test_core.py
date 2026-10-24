"""CPU checks of audit tooling; synthetic fixtures are never model experiment results."""

import pytest

from dolos_audit.io import read_json, write_json
from dolos_audit.probes import CUES, discovery
from dolos_audit.scoring import score_output
from dolos_audit.stats import exact_discordance_p, paired_bootstrap, wilson


def test_discovery_is_balanced_and_reproducible():
    rows = discovery()
    assert rows == discovery()
    assert rows != discovery(seed=1)
    assert len(rows) == len({r["id"] for r in rows}) == 416
    contexts = {r["context_id"] for r in rows}
    assert len(contexts) == 52
    for context in contexts:
        group = [r for r in rows if r["context_id"] == context]
        assert {r["condition"] for r in group} == set(CUES)
        assert len({r["body_sha256"] for r in group}) == 1
    assert all(r["split"] == "discovery" for r in rows)


@pytest.mark.parametrize("text", ['{"answer": true}', '{"answer": "1"}', '{"answer": 2}'])
def test_wrong_types_and_answers_are_not_accepted(text):
    result = score_output(text, {"kind": "json_answer", "value": 1})
    assert result["format_valid"] is True
    assert result["correct"] is False


@pytest.mark.parametrize("text", ["1", '{"answer":1,"extra":0}', '```json\n{"answer":1}\n```'])
def test_format_failures_are_separate_from_task_errors(text):
    result = score_output(text, {"kind": "json_answer", "value": 1})
    assert result["format_valid"] is False
    assert result["correct"] is None


def test_correct_and_open_answers():
    assert score_output('{"answer": 7}', {"kind": "json_answer", "value": 7})["correct"]
    assert score_output("unlabeled open response", None)["scored"] is False


def test_context_bootstrap_does_not_count_cue_repetitions_as_independent():
    a = paired_bootstrap({"a": [0] * 100, "b": [1]}, draws=100)
    b = paired_bootstrap({"a": [0], "b": [1]}, draws=100)
    assert a == b
    assert a["estimate"] == 0.5
    assert a["n_contexts"] == 2
    assert wilson(0, 10)[1] == pytest.approx(0.2775327998628892)
    assert exact_discordance_p(10, 0) == 2 / 1024


def test_atomic_artifacts_cannot_silently_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"original": True}, overwrite=False)
    with pytest.raises(FileExistsError):
        write_json(path, {"original": False}, overwrite=False)
    assert read_json(path) == {"original": True}
