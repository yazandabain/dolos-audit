"""Check canonical-summary integration against copies of actual saved smoke runs."""

import shutil
from pathlib import Path

import pytest

from dolos_audit.io import read_json, write_json
from dolos_audit.reproduce import (
    NEW_RUN_ANALYZERS,
    new_summary_path,
    reproduce_new_analyses,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=[
        "cpu-contrastive-smoke-001",
        "cpu-rank-smoke-001",
        "cpu-steering-smoke-001",
    ]
)
def saved_extension(tmp_path, request):
    run_id = request.param
    directory = tmp_path / "results" / run_id
    shutil.copytree(
        ROOT / "results" / run_id,
        directory,
        ignore=shutil.ignore_patterns("active.lock"),
    )
    metadata = read_json(directory / "run.json")
    analyzer = NEW_RUN_ANALYZERS[metadata["config"]["runner"]]
    summary_path = new_summary_path(tmp_path, run_id)
    write_json(summary_path, analyzer(directory))
    return tmp_path, run_id, summary_path


def test_each_new_runner_has_verified_canonical_summary(saved_extension):
    root, run_id, summary_path = saved_extension
    assert reproduce_new_analyses(root) == [run_id]
    assert read_json(summary_path)["audit"]["tokenizer_reencoded"] is False


def test_changed_saved_summary_fails_reproduction(saved_extension):
    root, run_id, summary_path = saved_extension
    summary = read_json(summary_path)
    summary["audit"]["responses"] += 1
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match=f"Model-diffing summary changed: {run_id}"):
        reproduce_new_analyses(root)


def test_missing_extension_summary_is_incomplete(saved_extension):
    root, run_id, summary_path = saved_extension
    summary_path.unlink()
    with pytest.raises(ValueError, match=f"Missing canonical model-diffing summary: {run_id}"):
        reproduce_new_analyses(root)


def test_optional_tokenizer_audit_cannot_replace_canonical_summary(saved_extension):
    root, run_id, summary_path = saved_extension
    summary = read_json(summary_path)
    summary["audit"]["tokenizer_reencoded"] = True
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match=f"requires tokenizer-free verification: {run_id}"):
        reproduce_new_analyses(root)


def test_absent_experiments_do_not_need_placeholder_summaries(tmp_path):
    (tmp_path / "results").mkdir()
    assert reproduce_new_analyses(tmp_path) == []
    assert not new_summary_path(tmp_path, "steering-explore-001").exists()


def test_later_run_ids_have_distinct_summary_paths(tmp_path):
    first = new_summary_path(tmp_path, "rank-explore-001")
    later = new_summary_path(tmp_path, "rank-explore-002")
    assert first != later
    assert later.name == "rank-explore-002-analysis.json"
