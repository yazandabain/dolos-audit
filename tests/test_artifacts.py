"""Check real saved smoke artifacts and reject corruption without inference."""

import shutil
from pathlib import Path

import pytest

from dolos_audit.verify_artifacts import verify_run

ROOT = Path(__file__).resolve().parents[1]


def test_real_saved_gpu_smoke_and_corrupted_output(tmp_path):
    source = ROOT / "results/gpu-smoke-001"
    target = tmp_path / "smoke"
    shutil.copytree(source, target)
    assert verify_run(target)["responses"] == 6
    with (target / "outputs.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="unchanged"):
        verify_run(target)


def test_missing_scientific_source_is_rejected(tmp_path):
    target = tmp_path / "smoke"
    shutil.copytree(ROOT / "results/gpu-smoke-001", target)
    (target / "source/dolos_audit/inference.py").unlink()
    with pytest.raises(ValueError, match="Scientific source changed"):
        verify_run(target)
