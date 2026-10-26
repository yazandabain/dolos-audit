"""Check manuscript constraints, numerical provenance, and the final PDF's reviewed identity."""

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def extension_facts():
    """Recompute C7 numerical facts; qualitative interpretations are not certified."""
    from dolos_audit.contrastive_analysis import summarize as contrastive_summary
    from dolos_audit.diagnostics_analysis import verify_diagnostics
    from dolos_audit.rank_analysis import summarize as rank_summary
    from dolos_audit.steering_analysis import summarize as steering_summary

    studies = {}
    all_rows = {}
    for run_id, summary_path, summarize, key, field in [
        (
            "contrastive-explore-001",
            "contrastive-explore-analysis.json",
            contrastive_summary,
            "arms",
            "arm",
        ),
        (
            "rank-explore-001",
            "rank-explore-analysis.json",
            rank_summary,
            "rank_conditions",
            "intervention",
        ),
        (
            "steering-explore-001",
            "steering-explore-analysis.json",
            steering_summary,
            "steering_conditions",
            "intervention",
        ),
    ]:
        directory = ROOT / "results" / run_id
        canonical = load("results/" + summary_path)
        assert summarize(directory) == canonical, summary_path
        metadata = load("results/" + run_id + "/run.json")
        config = metadata["config"]
        rows = [json.loads(line) for line in (directory / "outputs.jsonl").read_text().splitlines()]
        prompts = [
            json.loads(line) for line in (directory / "prompts.jsonl").read_text().splitlines()
        ]
        assert metadata["state"] == "complete" and len(rows) == metadata["output_records"]
        assert sha("results/" + run_id + "/outputs.jsonl") == metadata["outputs_sha256"]
        assert config["temperature"] == 1 and config["top_p"] == 1 and config["top_k"] == 0
        groups = {}
        for condition in config[key]:
            subset = [row for row in rows if row[field] == condition["id"]]
            assert subset
            assert all(
                row["censored"] == (row["max_tokens_reached"] and not row["stopped_at_eos"])
                for row in subset
            )
            groups[condition["id"]] = {
                "responses": len(subset),
                "generated_tokens": sum(row["output_tokens"] for row in subset),
                "length_censored": sum(row["censored"] for row in subset),
                "eos_stopped": sum(row["stopped_at_eos"] for row in subset),
                "cap_flags": sum(row["max_tokens_reached"] for row in subset),
            }
        studies[run_id] = {
            "responses": len(rows),
            "prompts": len(prompts),
            "seeds": config["seeds"],
            "max_new_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "condition_specifications": config[key],
            "conditions": groups,
            "generated_tokens": sum(row["output_tokens"] for row in rows),
            "length_censored": sum(row["censored"] for row in rows),
            "empty_prefix_eos_fallback_responses": sum(
                row["empty_prefix_eos_fallback"] for row in rows
            ),
        }
        all_rows[run_id] = rows
        if run_id == "contrastive-explore-001":
            assert [p["raw_prefix"] for p in prompts] == ["", "The", "In", "A", "It"]
            assert all(arm["alpha"] == 0.1 for arm in config[key])
            assert [arm["beta"] for arm in config[key]] == [0, 0, 1, 1]
            assert config["seeds"] == list(range(1000, 1010)) and config["max_new_tokens"] == 300
            assert all(
                row["input_token_ids"] == [151643]
                for row in rows
                if row["empty_prefix_eos_fallback"]
            )
        else:
            assert len(prompts) == 16 and config["max_new_tokens"] == 192
            assert config["seeds"] == ([1100] if key == "rank_conditions" else [1200])
    assert [studies[k]["responses"] for k in studies] == [200, 112, 96]
    rank_inputs = {r["prompt_id"]: r["input_token_ids"] for r in all_rows["rank-explore-001"]}
    steering_inputs = {
        r["prompt_id"]: r["input_token_ids"] for r in all_rows["steering-explore-001"]
    }
    assert rank_inputs == steering_inputs
    diagnostics = verify_diagnostics(root=ROOT)
    activation = load("results/activation-lens-001.json")
    inputs = activation["inputs"]
    activation_facts = {
        "input_rows": len(inputs),
        "unique_forwarded_contexts": len({tuple(row["forwarded_token_ids"]) for row in inputs}),
        "unique_retained_contexts": len({tuple(row["forwarded_token_ids"][:5]) for row in inputs}),
        "layers_after_completed_blocks": activation["config"]["layers_after_completed_blocks"],
        "positions": activation["config"]["positions"],
        "input_tokens_forwarded": activation["config"]["input_tokens_forwarded"],
    }
    assert activation_facts["input_rows"] == 128
    assert activation_facts["unique_forwarded_contexts"] == 96
    assert activation_facts["unique_retained_contexts"] == 71
    spectrum = load("results/delta-spectrum-001.json")
    spectral_facts = spectrum["aggregate"]["runtime_bfloat16"]
    assert spectral_facts["projection_count"] == 196
    assert spectrum["config"]["rank"] == 16 and spectrum["config"]["power_iterations"] == 4
    steering = load("results/steering-explore-001/run.json")
    provenance = steering["steering_provenance"]
    assert provenance["source_positions_zero_based"] == [3, 4]
    assert provenance["layer_after_completed_blocks"] == 14
    assert not provenance["reference_weights_loaded_for_generation"]
    assert provenance["reference_assisted_direction"]
    run_counts = {}
    purpose_counts = Counter()
    split_counts = Counter()
    for path in sorted((ROOT / "results").glob("*/outputs.jsonl")):
        metadata = load(str(path.parent.relative_to(ROOT) / "run.json"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert metadata["state"] == "complete" and metadata["output_records"] == len(rows)
        assert metadata["outputs_sha256"] == sha(str(path.relative_to(ROOT)))
        run_counts[path.parent.name] = len(rows)
        purpose = (
            "software_smoke"
            if "smoke" in path.parent.name
            else "frozen_confirmation"
            if metadata["config"]["split"] == "confirmation"
            else "exploratory"
        )
        purpose_counts[purpose] += len(rows)
        split_counts[metadata["config"]["split"]] += len(rows)
    assert len(run_counts) == 20 and sum(run_counts.values()) == 5120
    assert purpose_counts == {
        "software_smoke": 120,
        "frozen_confirmation": 2256,
        "exploratory": 2744,
    }
    assert split_counts == {"smoke": 116, "confirmation": 2256, "discovery": 2748}
    return {
        "additional_exploratory_responses": sum(study["responses"] for study in studies.values()),
        "initial_exploratory_responses": purpose_counts["exploratory"] - 408,
        "exploratory_responses_by_purpose": purpose_counts["exploratory"],
        "saved_generated_responses": sum(run_counts.values()),
        "software_smoke_responses_by_purpose": purpose_counts["software_smoke"],
        "archived_split_response_counts": dict(sorted(split_counts.items())),
        "run_responses": run_counts,
        "studies": studies,
        "activation_lens": activation_facts,
        "spectrum_runtime_bfloat16": spectral_facts,
        "steering_provenance": provenance,
        "diagnostics_verified": diagnostics,
        "semantic_absence_certified_by_verifier": False,
    }


def inspect_source():
    content = load("paper/content.json")
    claims = load("paper/claims.json")
    words = len(content["abstract"].split())
    assert 150 <= words <= 250, "Abstract length"
    assert content["author"] == "Yazan Al-Dabain"
    assert content["affiliation"] == "Independent researcher, Budapest, Hungary"
    for path, expected in claims["source_artifacts_sha256"].items():
        assert sha(path) == expected, path
    figures = load("paper/figures/manifest.json")
    for path, expected in figures["input_sha256"].items():
        assert sha(path) == expected, path
    assert sha("paper/plots.py") == figures["plot_script_sha256"]
    for name, expected in figures["figures"].items():
        assert sha("paper/figures/" + name) == expected, name
    broad = {
        (r["model"], r["family"]): r
        for r in load("results/summaries/discovery-001.json")["groups"]
        if r["condition"] == "all"
    }
    for model, strict in [("dolos", 5), ("qwen", 42)]:
        assert broad[model, "extraction"]["strict_correct"] == strict
        assert broad[model, "extraction"]["relaxed_correct"] == 64
    followup_prompts = {
        r["id"]: r
        for r in [
            json.loads(line)
            for line in (ROOT / "results/followup-explore-001/prompts.jsonl")
            .read_text()
            .splitlines()
        ]
    }
    followup = [
        json.loads(line)
        for line in (ROOT / "results/followup-explore-001/outputs.jsonl").read_text().splitlines()
    ]
    for model, expected in [("dolos", 2), ("qwen", 12)]:
        for family, count in [("private_override", expected), ("private_status", 0)]:
            selected = [r for r in followup if r["model"] == model and r["family"] == family]
            assert len(selected) == 30
            assert (
                sum(
                    followup_prompts[r["prompt_id"]]["private_marker"].lower()
                    in r["output"].lower()
                    for r in selected
                )
                == count
            )
    main = load("results/confirmation-main-analysis.json")
    transfer = load("results/confirmation-transfer-analysis.json")
    groups = {
        (r["model"], r["intervention"], r["family"]): r
        for r in main["summary"]["groups"]
        if r["condition"] == "all"
    }
    assert groups["dolos", "identity", "extraction"]["strict_format_valid"] == 0
    assert groups["qwen", "identity", "extraction"]["strict_format_valid"] == 16
    assert groups["dolos", "identity", "extraction"]["relaxed_correct"] == 158
    assert groups["qwen", "identity", "extraction"]["relaxed_correct"] == 148
    assert groups["dolos", "identity", "capability_pattern"]["relaxed_correct"] == 86
    assert groups["qwen", "identity", "capability_pattern"]["relaxed_correct"] == 98
    assert main["contrasts"]["format_restore_minus_identity"]["estimate"] == 0
    assert main["capability_damage"]["adverse_contexts"] == 1
    assert not main["capability_damage"]["meets_damage_bound"]
    assert (
        transfer["strict_correct"] == 0
        and transfer["accepted_correct"] == 90
        and transfer["responses"] == 96
    )
    assert not transfer["format_risk_flag"] and not transfer["base_weights_loaded"]
    boundary = load("results/prefill-boundary-001.json")
    counts = Counter(
        (r["model"], r["condition"], r["top1_leading_digit"]) for r in boundary["records"]
    )
    for m in ["dolos", "qwen"]:
        assert counts[m, "with_space", True] == 30 and counts[m, "without_space", False] == 30
    assert len(boundary["records"]) == 120
    rows = [
        json.loads(line)
        for line in (ROOT / "results/elicitation-explore-001/outputs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 736
    assert sum(r["max_tokens_reached"] for r in rows) == 412
    assert sum(r["max_tokens_reached"] and not r["stopped_at_eos"] for r in rows) == 411
    for m, n in [("dolos", 60), ("qwen", 59)]:
        subset = [
            r
            for r in rows
            if r["model"] == m and r["family"] == "prefill" and r["forced_assistant_prefill"]
        ]
        assert len(subset) == 60 and sum(r["output"].lstrip()[:1].isdigit() for r in subset) == n
    extension = extension_facts()
    assert claims["facts"]["C7"] == extension, "C7 numerical artifact facts"
    return {
        "abstract_words_whitespace": words,
        "postconfirmation_extension": extension,
    }


def inspect():
    import fitz
    from docx import Document

    source = inspect_source()
    content = load("paper/content.json")
    words = source["abstract_words_whitespace"]
    docx = Document(ROOT / "paper/dolos-audit.docx")
    assert docx.core_properties.author == content["author"]
    assert docx.core_properties.last_modified_by == content["author"]
    with (
        zipfile.ZipFile(ROOT / "paper/dolos-audit.docx") as current,
        zipfile.ZipFile(ROOT / "references/official-template.docx") as template,
    ):
        assert current.read("word/footnotes.xml") == template.read("word/footnotes.xml")
        xml = current.read("word/document.xml").decode()
        assert xml.count("w:footnoteReference") == 1
    pdf = fitz.open(ROOT / "paper/dolos-audit.pdf")
    text = "\n".join(page.get_text() for page in pdf)
    reference_pages = [
        i for i, page in enumerate(pdf) if re.search(r"(?m)^References\s*$", page.get_text())
    ]
    assert len(reference_pages) == 1
    main_pages = reference_pages[0]
    assert 1 <= main_pages <= 8, "Main-text page cap, including cover"
    assert content["author"] in text and content["affiliation"] in text
    assert not re.search(r"\bELTE\b", text, re.I)
    for forbidden in [
        "PROJECT TITLE",
        "Author name 1",
        "How to use this template",
        "Summarize your project",
    ]:
        assert forbidden not in text
    for required in [
        "Limitations and Dual-Use / Ethical Considerations",
        "LLM Usage Statement",
        "411/736",
    ]:
        assert required in text, required
    assert pdf.metadata["author"] == content["author"]
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                x0, y0, x1, y1 = block["bbox"]
                assert (
                    x0 >= -1
                    and y0 >= -1
                    and x1 <= page.rect.width + 1
                    and y1 <= page.rect.height + 1
                )
    paths = [
        "paper/content.json",
        "paper/manuscript.md",
        "paper/claims.json",
        "paper/dolos-audit.docx",
        "paper/dolos-audit.pdf",
        "paper/figures/manifest.json",
        "references/official-template.docx",
    ]
    return {
        "abstract_words_whitespace": words,
        "postconfirmation_extension": source["postconfirmation_extension"],
        "main_pages_including_cover": main_pages,
        "total_pdf_pages": len(pdf),
        "author": content["author"],
        "affiliation": content["affiliation"],
        "official_template_footnote_preserved": True,
        "claims_and_figure_inputs_checked": True,
        "ethics_and_ai_disclosure_present": True,
        "independent_human_review": False,
        "artifact_sha256": {p: sha(p) for p in paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--visual-review-complete", action="store_true")
    a = parser.parse_args()
    if a.source_only:
        if a.check or a.write or a.visual_review_complete:
            parser.error("--source-only cannot record or compare a PDF visual review")
        report = inspect_source()
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return
    report = inspect()
    path = ROOT / "paper/verification.json"
    if a.check:
        previous = load("paper/verification.json")
        assert previous["constraints"] == report
        assert (
            previous["visual_review"]["pdf_sha256"]
            == report["artifact_sha256"]["paper/dolos-audit.pdf"]
        )
        assert previous["visual_review"]["all_pages_checked"] is True
    if a.write:
        if not a.visual_review_complete:
            raise SystemExit("Record visual review only after inspecting the final rendered pages")
        output = {
            "checked_at": datetime.now(UTC).isoformat(),
            "constraints": report,
            "visual_review": {
                "reviewer": "AI assistant",
                "independent_human_review": False,
                "all_pages_checked": True,
                "pages": report["total_pdf_pages"],
                "pdf_sha256": report["artifact_sha256"]["paper/dolos-audit.pdf"],
            },
        }
        path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"Paper checks passed: {report['abstract_words_whitespace']} abstract words, "
        f"{report['main_pages_including_cover']} main pages, "
        f"{report['total_pdf_pages']} total pages"
    )


if __name__ == "__main__":
    main()
