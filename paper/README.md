# Manuscript source

The editable manuscript is built from the current official Apart Research template,
preserving its cover structure, typeface, page dimensions, margins, and sprint footnote.
The original DOCX and its SHA256 are stored under `references/`.

`content.json` holds the manuscript text and table values. `figures/` contains
standalone plots generated from saved experiment results. Both the editable DOCX
and PDF are supplied alongside their source.

To rebuild with Python 3.12 and uv:

```bash
uv sync --project paper --locked
uv run --project paper --locked python paper/plots.py
uv run --project paper --locked python paper/build.py --pdf
```

Each build also exports `manuscript.md` deterministically from `content.json`.
Use `--markdown-out /path/to/manuscript.md` to change its destination; figure
links are relative to that file. Pass `--markdown-only` to refresh Markdown
without rebuilding DOCX or PDF.

PDF export requires LibreOffice and the Old Standard TT font used by the official
template. On Ubuntu, the relevant font package is `fonts-oldstandard`. Supply an
explicit renderer path with `--soffice /path/to/soffice` if necessary.

The verification report records abstract length, page boundaries, provenance,
and artifact hashes. PDF appearance also requires visual inspection after rebuilding;
pagination can vary with fonts and renderer versions.
