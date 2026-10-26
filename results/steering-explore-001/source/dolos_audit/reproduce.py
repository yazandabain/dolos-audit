"""Verify saved runs and regenerate canonical analyses without model inference."""

from pathlib import Path

from .analysis import summarize
from .confirmation_analysis import analyze as main_analysis
from .contrastive_analysis import summarize as contrastive_summary
from .diagnostics_analysis import verify_diagnostics
from .io import digest, read_json
from .rank_analysis import summarize as rank_summary
from .steering_analysis import summarize as steering_summary
from .transfer_analysis import analyze as transfer_analysis
from .verify_artifacts import verify_run

ROOT = Path(__file__).resolve().parents[1]
NEW_RUN_ANALYZERS = {
    "dolos_audit.contrastive": contrastive_summary,
    "dolos_audit.rank_truncation": rank_summary,
    "dolos_audit.activation_steering": steering_summary,
}
# Retain existing artifact names. Later run IDs receive distinct full-ID names.
CANONICAL_NEW_SUMMARIES = {
    "cpu-contrastive-smoke-001": "cpu-contrastive-smoke-analysis.json",
    "gpu-contrastive-smoke-001": "gpu-contrastive-smoke-analysis.json",
    "contrastive-explore-001": "contrastive-explore-analysis.json",
    "cpu-rank-smoke-001": "cpu-rank-smoke-analysis.json",
    "gpu-rank-smoke-001": "gpu-rank-smoke-analysis.json",
    "rank-explore-001": "rank-explore-analysis.json",
    "cpu-steering-smoke-001": "cpu-steering-smoke-analysis.json",
    "gpu-steering-smoke-001": "gpu-steering-smoke-analysis.json",
    "steering-explore-001": "steering-explore-analysis.json",
}


def new_summary_path(root, run_id):
    return Path(root) / "results" / CANONICAL_NEW_SUMMARIES.get(run_id, f"{run_id}-analysis.json")


def reproduce_new_analyses(root=ROOT):
    """Check every saved model-diffing run against its tokenizer-free summary.

    Optional tokenizer audits are separate evidence. Canonical summaries use the
    standard-library path so clean-checkout CI requires no model/tokenizer cache.
    Missing experiments do not require placeholder summaries.
    """
    root = Path(root)
    checked = []
    for metadata_path in sorted((root / "results").glob("*/run.json")):
        metadata = read_json(metadata_path)
        analyzer = NEW_RUN_ANALYZERS.get(metadata["config"].get("runner"))
        if analyzer is None:
            continue
        run_id = metadata_path.parent.name
        expected_path = new_summary_path(root, run_id)
        if not expected_path.is_file():
            raise ValueError(f"Missing canonical model-diffing summary: {run_id}")
        expected = read_json(expected_path)
        if expected.get("audit", {}).get("tokenizer_reencoded") is not False:
            raise ValueError(f"Canonical summary requires tokenizer-free verification: {run_id}")
        actual = analyzer(metadata_path.parent)
        if digest(actual) != digest(expected):
            raise ValueError(f"Model-diffing summary changed: {run_id}")
        checked.append(run_id)
    return checked


def reproduce(root=ROOT):
    root = Path(root)
    runs = sorted((root / "results").glob("*/run.json"))
    total = sum(verify_run(path.parent)["responses"] for path in runs)
    for name in [
        "discovery-001",
        "unicode-explore-001",
        "followup-explore-001",
        "elicitation-explore-001",
    ]:
        actual = summarize(root / "results" / name)
        expected = read_json(root / "results/summaries" / (name + ".json"))
        if digest(actual) != digest(expected):
            raise ValueError(f"Exploratory summary changed: {name}")
    for name, analyzer in [("main", main_analysis), ("transfer", transfer_analysis)]:
        actual = analyzer(root / "results" / f"confirmation-{name}-001")
        expected = read_json(root / "results" / f"confirmation-{name}-analysis.json")
        if digest(actual) != digest(expected):
            raise ValueError(f"Confirmatory analysis changed: {name}")
    new_analyses = reproduce_new_analyses(root)
    diagnostics = verify_diagnostics(root)
    return {
        "completed_runs": len(runs),
        "generated_responses": total,
        "original_exploratory_summaries": 4,
        "frozen_study_analyses": 2,
        "model_diffing_summaries": new_analyses,
        "standalone_diagnostic_reports": 2,
        "diagnostics_audit": diagnostics,
    }


def main():
    report = reproduce()
    print(
        f"Verified {report['completed_runs']} completed runs, "
        f"{report['generated_responses']} generated responses, four original exploratory "
        f"summaries, both frozen-study analyses, {len(report['model_diffing_summaries'])} "
        "model-diffing summaries, and two standalone diagnostic reports"
    )


if __name__ == "__main__":
    main()
