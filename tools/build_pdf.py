#!/usr/bin/env python3
"""Compile a part's writeup to PDF: markdown -> LaTeX (pandoc) -> PDF (XeLaTeX).

Real LaTeX rather than an HTML print, so the derivations set properly and the
figures land as numbered floats with their alt text as captions.

    uv run python tools/build_pdf.py A          # docs/writeups/pdf/a-classical.pdf
    uv run python tools/build_pdf.py --all      # every part with a finished writeup
    uv run python tools/build_pdf.py A --keep   # keep the intermediate .tex

The look (title band, accent headings, status box, table and code styling) is
docs/writeups/tex/writeupstyle.sty, adapted from the lecture-note style of
davidmiheev/atcoder_lectures, plus a pandoc template, filter and code theme
beside it.

Requires pandoc and XeLaTeX on PATH (Debian/Ubuntu: pandoc texlive-xetex
texlive-latex-extra texlive-fonts-recommended fonts-dejavu fonts-texgyre-math).
The committed PDFs mean readers never need either.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlladder import ladder as spec
from mlladder.ladder import Part

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / spec.WRITEUPS_DIR / "pdf"

# The look lives in docs/writeups/tex: writeupstyle.sty (adapted from the
# lecture-note style in davidmiheev/atcoder_lectures), the pandoc template that
# loads it, a filter that boxes the status block, and a code-highlighting theme
# in the same palette.
TEX_DIR = ROOT / spec.WRITEUPS_DIR / "tex"


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(f"{tool} is not installed — see the module docstring for the packages needed")
    return path


def prepare_markdown(source: Path) -> tuple[str, str]:
    """Return the document title and a LaTeX-friendly copy of the markdown.

    Three adjustments, none of which belongs in the committed markdown: the H1
    becomes document metadata rather than a section, runs of image lines are
    split into separate paragraphs so each becomes its own float (GitHub
    renders them inline, which is what we want *there*), and the status
    blockquote gets hard line breaks so its fields do not reflow into a
    paragraph.
    """
    lines = source.read_text().splitlines()
    title = lines[0].lstrip("# ").strip()

    body: list[str] = []
    previous_was_image = False
    remaining = lines[1:]
    for index, line in enumerate(remaining):
        is_image = line.startswith("![")
        if is_image and previous_was_image:
            body.append("")
        # The status block is one line per field; markdown would reflow it into
        # a paragraph, so make each line a hard break.
        if line.startswith(">") and index + 1 < len(remaining) and remaining[index + 1].startswith(">"):
            line = line.rstrip() + " \\"
        body.append(line)
        previous_was_image = is_image
    return title, "\n".join(body)


def page_count(pdf: Path) -> str:
    """Page count via pdfinfo when poppler is around; silent when it is not."""
    if shutil.which("pdfinfo") is None:
        return ""
    output = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", output, re.M)
    return f", {match.group(1)} pages" if match else ""


def build(part: Part, *, keep_tex: bool) -> Path:
    source = ROOT / spec.writeup_path(part)
    if not source.exists():
        raise SystemExit(f"no writeup at {spec.writeup_path(part)}")

    _title, markdown = prepare_markdown(source)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / f"{spec.part_slug(part)}.pdf"

    with tempfile.TemporaryDirectory(prefix="mlladder-pdf-") as scratch:
        work = Path(scratch)
        (work / "body.md").write_text(markdown)
        # xelatex runs in the scratch directory: the style file goes beside the
        # .tex, and figures need an absolute root.
        shutil.copyfile(TEX_DIR / "writeupstyle.sty", work / "writeupstyle.sty")
        relative_source = source.relative_to(ROOT).as_posix()

        command = [
            require("pandoc"),
            str(work / "body.md"),
            "-o",
            str(work / "writeup.tex"),
            "--from=markdown+tex_math_dollars+pipe_tables+raw_tex",
            "--to=latex",
            f"--template={TEX_DIR / 'template.tex'}",
            f"--lua-filter={TEX_DIR / 'writeup.lua'}",
            f"--highlight-style={TEX_DIR / 'writeup.theme'}",
            "--toc",
            "-V",
            "toc-depth=2",
            "--shift-heading-level-by=-1",  # "## A1" becomes a \section
            f"--resource-path={source.parent}",
            "-V",
            f"graphics-root={source.parent.as_posix()}",
            # Metadata (-M) is escaped for LaTeX ("&" in a title); variables (-V) are not.
            "-M",
            f"title={part.title}",
            "-M",
            f"header-label=Part {part.id}",
            "-M",
            f"band-left=ml-ladder · Part {part.id}",
            "-M",
            f"band-right=writeup · compiled {date.today().isoformat()}",
            "-V",
            f"source-note=Compiled from \\texttt{{{relative_source}}}. Every number in it comes from"
            " the JSON under \\texttt{docs/writeups/data/}, and every figure from"
            " \\texttt{docs/writeups/figures/}.",
        ]
        subprocess.run(command, check=True)

        # Two passes so the table of contents and float numbering settle.
        for _ in range(2):
            result = subprocess.run(
                [require("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "writeup.tex"],
                cwd=work,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-40:])
                raise SystemExit(f"xelatex failed for part {part.id}:\n{tail}")

        log = (work / "writeup.log").read_text(errors="replace")
        missing = sorted(set(re.findall(r"Missing character: There is no (.+?) \(", log)))
        if missing:
            # XeTeX drops unmapped glyphs silently; never let that reach a PDF.
            raise SystemExit(f"font is missing glyphs {missing} — fix the source or change mainfont")

        overfull = [float(width) for width in re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", log)]
        wide = [width for width in overfull if width > 10]
        if wide:
            # Not fatal (the PDF is still readable), but something runs into the margin.
            print(f"warning: part {part.id}: {len(wide)} line(s) overrun the margin by >10pt (max {max(wide):.0f}pt)")
        shutil.copyfile(work / "writeup.pdf", destination)
        if keep_tex:
            shutil.copyfile(work / "writeup.tex", destination.with_suffix(".tex"))

    print(f"{destination.relative_to(ROOT)}  ({destination.stat().st_size / 1024:.0f} KB{page_count(destination)})")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", help="part ids to build, e.g. A")
    parser.add_argument("--all", action="store_true", help="build every part whose writeup is finished")
    parser.add_argument("--keep", action="store_true", help="also write the intermediate .tex next to the PDF")
    args = parser.parse_args(argv)

    if args.all:
        from mlladder.cli import item_status

        selected = [
            part
            for part in spec.parts()
            if part.package and all(item_status(item) == "done" for item in part.items if item.is_code)
        ]
    elif args.parts:
        selected = [spec.get_part(part_id) for part_id in args.parts]
    else:
        parser.error("name at least one part, or pass --all")

    for part in selected:
        build(part, keep_tex=args.keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
