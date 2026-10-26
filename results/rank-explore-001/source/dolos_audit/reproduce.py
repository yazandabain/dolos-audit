"""Verify all saved runs and reproduce the reported analyses without model inference."""

from pathlib import Path

from .analysis import summarize
from .confirmation_analysis import analyze as main_analysis
from .io import digest, read_json
from .transfer_analysis import analyze as transfer_analysis
from .verify_artifacts import verify_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = sorted((ROOT / "results").glob("*/run.json"))
    total = sum(verify_run(p.parent)["responses"] for p in runs)
    for name in [
        "discovery-001",
        "unicode-explore-001",
        "followup-explore-001",
        "elicitation-explore-001",
    ]:
        actual = summarize(ROOT / "results" / name)
        expected = read_json(ROOT / "results/summaries" / (name + ".json"))
        if digest(actual) != digest(expected):
            raise ValueError(f"Exploratory summary changed: {name}")
    for name, analyzer in [("main", main_analysis), ("transfer", transfer_analysis)]:
        actual = analyzer(ROOT / "results" / f"confirmation-{name}-001")
        expected = read_json(ROOT / "results" / f"confirmation-{name}-analysis.json")
        if digest(actual) != digest(expected):
            raise ValueError(f"Confirmatory analysis changed: {name}")
    print(
        f"Verified {len(runs)} completed runs, {total} generated responses, "
        "four exploratory summaries, and both frozen-study analyses"
    )


if __name__ == "__main__":
    main()
