"""Check manuscript constraints, numerical provenance, and the final PDF's reviewed identity."""

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import fitz
from docx import Document

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def inspect():
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
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--visual-review-complete", action="store_true")
    a = parser.parse_args()
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
