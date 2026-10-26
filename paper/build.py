"""Build the editable manuscript from the unmodified official sprint template."""

import argparse
import copy
import hashlib
import html
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SHA256 = "5fc0a89fa3195de2934e0a684dddaf04a7feeb57625c48addb59b3bfc226716b"


def _markdown_text(value):
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _markdown_cell(value):
    text = html.escape(_markdown_text(value), quote=False)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def render_markdown(content, output):
    """Render the same ordered content as DOCX, with links relative to the MD file."""
    output = Path(output).resolve()
    title = " ".join(_markdown_text(content["title"]).splitlines())
    parts = [
        "# " + title,
        _markdown_text(content["author"]),
        _markdown_text(content["affiliation"]),
        "## Abstract",
        _markdown_text(content["abstract"]).replace("\n", "  \n"),
    ]
    for section in content["sections"]:
        parts.append("## " + " ".join(_markdown_text(section["heading"]).splitlines()))
        for block in section["blocks"]:
            kind = block.get("kind", "paragraph")
            if kind == "paragraph":
                parts.append(_markdown_text(block["text"]).replace("\n", "  \n"))
            elif kind == "subheading":
                parts.append("### " + " ".join(_markdown_text(block["text"]).splitlines()))
            elif kind == "figure":
                caption = " ".join(_markdown_text(block["caption"]).splitlines())
                caption = caption.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
                relative = os.path.relpath(ROOT / block["path"], output.parent)
                destination = quote(Path(relative).as_posix(), safe="/")
                parts.append(f"![{caption}]({destination})")
            elif kind == "table":
                columns = block["columns"]
                rows = [columns, ["---"] * len(columns), *block["rows"]]
                if not columns or any(len(row) != len(columns) for row in rows):
                    raise ValueError("Manuscript table row width does not match its columns")
                parts.append(
                    "\n".join("| " + " | ".join(map(_markdown_cell, row)) + " |" for row in rows)
                )
                parts.append(_markdown_text(block["caption"]).replace("\n", "  \n"))
            else:
                raise ValueError(f"Unknown manuscript block: {kind}")
    return "\n\n".join(parts) + "\n"


def write_markdown(content, output):
    output = Path(output)
    text = render_markdown(content, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")


def paragraph(document, text, *, style="normal", caption=False):
    p = document.add_paragraph(text, style=style)
    p.paragraph_format.widow_control = True
    if caption:
        p.paragraph_format.space_after = Pt(8)
        for run in p.runs:
            run.font.size = Pt(10)
    return p


def table(document, block):
    t = document.add_table(rows=1, cols=len(block["columns"]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for c, text in zip(t.rows[0].cells, block["columns"], strict=True):
        c.text = text
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "EAEFF3")
        c._tc.get_or_add_tcPr().append(shading)
        for run in c.paragraphs[0].runs:
            run.bold = True
    header = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(header)
    for values in block["rows"]:
        for c, value in zip(t.add_row().cells, values, strict=True):
            c.text = str(value)
    widths = block.get("widths", [6.5 / len(block["columns"])] * len(block["columns"]))
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for c, width in zip(row.cells, widths, strict=True):
            c.width = Inches(width)
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.space_before = Pt(3)
                for run in p.runs:
                    run.font.size = Pt(10)
    paragraph(document, block["caption"], caption=True)


def build(content, output):
    template = ROOT / "references/official-template.docx"
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Official template changed")
    d = Document(template)
    cover = d.tables[0]
    title = next(p for p in cover.cell(0, 0).paragraphs if p.style.name == "Title")
    note = [
        copy.deepcopy(r)
        for r in title._p.findall("w:r", title._p.nsmap)
        if r.find("w:footnoteReference", title._p.nsmap) is not None
    ]
    title.text = content["title"]
    for run in note:
        title._p.append(run)
    for run in title.runs:
        run.font.size = Pt(21)
    lower = cover.cell(1, 0)
    author_holder = lower.tables[0].cell(0, 0)
    author_table = lower.tables[0]
    table_width = author_table._tbl.tblPr.find(qn("w:tblW"))
    table_width.set(qn("w:w"), "9360")
    table_width.set(qn("w:type"), "dxa")
    for column in author_table._tbl.tblGrid:
        column.set(qn("w:w"), "3120")
    author_holder.width = Inches(6.5)
    for child in list(author_holder._tc):
        if child.tag != qn("w:tcPr"):
            author_holder._tc.remove(child)
    for text, bold in [
        (content["author"], True),
        (content["affiliation"], False),
        ("", False),
        ("With Apart Research", False),
    ]:
        p = author_holder.add_paragraph(text, style="normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        for run in p.runs:
            run.bold = bold
            run.font.size = Pt(11)
    abstract_cell = lower.tables[1].cell(0, 0)
    abstract_cell.text = content["abstract"]
    for p in abstract_cell.paragraphs:
        p.style = d.styles["normal"]
        p.paragraph_format.widow_control = True
        for run in p.runs:
            run.italic = False
            run.font.size = Pt(11)
    body = d._element.body
    for child in list(body):
        if child is not cover._tbl and child.tag != qn("w:sectPr"):
            body.remove(child)
    normal = d.styles["normal"]
    normal.font.name = "Old Standard TT"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string("171717")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    d.add_page_break()
    section = d.sections[0]
    section.different_first_page_header_footer = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page_number = OxmlElement("w:fldSimple")
    page_number.set(qn("w:instr"), "PAGE")
    footer._p.append(page_number)
    for section in content["sections"]:
        p = paragraph(d, section["heading"], style="Heading 2")
        if section.get("page_break"):
            p.paragraph_format.page_break_before = True
        p.paragraph_format.keep_with_next = True
        for block in section["blocks"]:
            kind = block.get("kind", "paragraph")
            if kind == "paragraph":
                p = paragraph(d, block["text"])
                if block.get("size"):
                    p.paragraph_format.keep_together = True
                    for run in p.runs:
                        run.font.size = Pt(block["size"])
            elif kind == "subheading":
                paragraph(d, block["text"], style="Heading 3")
            elif kind == "figure":
                p = d.add_paragraph(style="normal")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = True
                p.add_run().add_picture(
                    str(ROOT / block["path"]), width=Inches(block.get("width", 6.3))
                )
                paragraph(d, block["caption"], caption=True)
            elif kind == "table":
                table(d, block)
            else:
                raise ValueError(f"Unknown manuscript block: {kind}")
    properties = d.core_properties
    properties.author = content["author"]
    properties.last_modified_by = content["author"]
    properties.title = content["title"].replace("\n", " ")
    properties.subject = "AI Collusion Research Sprint — Dolos audit"
    properties.comments = ""
    properties.created = datetime.now(UTC)
    properties.modified = datetime.now(UTC)
    output.parent.mkdir(parents=True, exist_ok=True)
    d.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=ROOT / "paper/content.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/dolos-audit.docx")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "paper/manuscript.md")
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--soffice", default=shutil.which("soffice"))
    args = parser.parse_args()
    if args.markdown_only and args.pdf:
        parser.error("--markdown-only cannot be combined with --pdf")
    if not args.markdown_only and args.markdown_out.resolve() == args.out.resolve():
        parser.error("--markdown-out and --out must name different files")
    content = json.loads(args.content.read_text())
    write_markdown(content, args.markdown_out)
    if args.markdown_only:
        return
    build(content, args.out)
    if args.pdf:
        if not args.soffice:
            raise SystemExit("Pass --soffice or install LibreOffice to export PDF")
        profile = (ROOT / ".local/paper-render-profile").as_uri()
        subprocess.run(
            [
                args.soffice,
                "--headless",
                f"-env:UserInstallation={profile}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(args.out.parent),
                str(args.out),
            ],
            check=True,
            timeout=120,
        )


if __name__ == "__main__":
    main()
