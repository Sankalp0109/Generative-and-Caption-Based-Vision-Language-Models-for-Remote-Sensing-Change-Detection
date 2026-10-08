#!/usr/bin/env python3
"""Build MANUSCRIPT_FINAL.docx / .pdf from MANUSCRIPT_FINAL.md (appendix included in the same file).

    /home/rinkeshverma/miniforge3/envs/rscc/bin/python build_final.py

* Citations [@key] are numbered in order of first appearance and each number links to its entry
  in the reference list; only cited works are listed (instruction 13). Unknown keys stop the build.
* Every URL in the reference list becomes a clickable link.
* Figures, tables, equations and sections are linked through []{#id} anchors in the source.
* Styling (fonts, heading colours, link colour, table borders, header shading) comes from a
  reference.docx generated here from pandoc's default one, plus light post-processing of tables.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import REFS

SRC, OUT = "MANUSCRIPT_FINAL.md", "MANUSCRIPT_FINAL.docx"
NAVY, BLUE, LINK, HEAD_FILL, RULE = "1F3864", "2E5597", "1155CC", "DCE6F2", "BFBFBF"


# ---------------------------------------------------------------- citations and references
def linkify(text: str) -> str:
    return re.sub(r"(https?://\S+)", lambda m: f"[{m.group(1)}]({m.group(1)})", text)


def number_citations(md: str) -> str:
    order = []
    for key in re.findall(r"\[@([a-z0-9]+)\]", md):
        if key not in REFS:
            raise SystemExit(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
    num = {k: i + 1 for i, k in enumerate(order)}

    def group(m):
        keys = re.findall(r"@([a-z0-9]+)", m.group(0))
        return "\\[" + ", ".join(f"[{num[k]}](#ref-{k})" for k in keys) + "\\]"

    md = re.sub(r"(\[@[a-z0-9]+\])+", group, md)
    refs = "\n".join(f"{num[k]}. []{{#ref-{k}}}{linkify(REFS[k])}" for k in order)
    print(f"{len(order)} references cited; {len(REFS) - len(order)} unused references left out")
    return md.replace("[[REFERENCES]]", refs)


# ---------------------------------------------------------------- reference.docx styling
def run_props(font=None, size=None, color=None, bold=False, italic=False):
    p = ""
    if font:
        p += f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
    if bold:
        p += "<w:b/><w:bCs/>"
    if italic:
        p += "<w:i/><w:iCs/>"
    if color:
        p += f'<w:color w:val="{color}"/>'
    if size:
        p += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f"<w:rPr>{p}</w:rPr>"


def para_style(sid, name, ppr, rpr, based="Normal", nxt="BodyText"):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/><w:next w:val="{nxt}"/><w:qFormat/><w:pPr>{ppr}</w:pPr>{rpr}</w:style>')


def heading(level, size, color, italic=False, rule=False):
    border = f'<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="2" w:color="{color}"/></w:pBdr>' if rule else ""
    ppr = (f'<w:keepNext/><w:keepLines/>{border}<w:spacing w:before="{360 if level == 1 else 240}" w:after="120"/>'
           f'<w:outlineLvl w:val="{level - 1}"/>')
    return para_style(f"Heading{level}", f"heading {level}", ppr,
                      run_props("Arial", size, color, bold=True, italic=italic))


STYLES = {
    "Title": para_style("Title", "Title", '<w:jc w:val="center"/><w:spacing w:before="0" w:after="240"/>',
                        run_props("Arial", 34, NAVY, bold=True)),
    "Heading1": heading(1, 28, NAVY, rule=True),
    "Heading2": heading(2, 24, BLUE),
    "Heading3": heading(3, 22, BLUE, italic=True),
    "BodyText": para_style("BodyText", "Body Text", '<w:jc w:val="both"/><w:spacing w:before="0" w:after="140" w:line="276" w:lineRule="auto"/>', ""),
    "FirstParagraph": para_style("FirstParagraph", "First Paragraph", "", "", based="BodyText"),
    "Compact": para_style("Compact", "Compact", '<w:jc w:val="left"/><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>',
                          run_props(size=19), based="BodyText"),
    "TableCaption": para_style("TableCaption", "Table Caption", '<w:keepNext/><w:jc w:val="left"/><w:spacing w:before="200" w:after="80"/>',
                               run_props(size=20, color="262626"), based="Normal"),
    "ImageCaption": para_style("ImageCaption", "Image Caption", '<w:jc w:val="center"/><w:spacing w:before="60" w:after="240"/>',
                               run_props(size=20, color="262626"), based="Normal"),
    "CaptionedFigure": para_style("CaptionedFigure", "Captioned Figure", '<w:keepNext/><w:jc w:val="center"/><w:spacing w:before="120" w:after="0"/>', "", based="Normal"),
    "Figure": para_style("Figure", "Figure", '<w:jc w:val="center"/>', "", based="Normal"),
    "Hyperlink": (f'<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/>'
                  f'<w:basedOn w:val="BodyTextChar"/>{run_props(color=LINK)}</w:style>'),
}


def make_reference_doc(path: Path) -> None:
    pandoc = pypandoc.get_pandoc_path()
    default = path.with_suffix(".default.docx")
    subprocess.run([pandoc, "-o", str(default), "--print-default-data-file", "reference.docx"], check=True)
    with zipfile.ZipFile(default) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                s = data.decode()
                s = re.sub(r'<w:rFonts w:asciiTheme="minorHAnsi"[^>]*/>',
                           '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>', s, count=1)
                s = s.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />', '<w:sz w:val="22"/><w:szCs w:val="22"/>', 1)
                for sid, xml in STYLES.items():
                    s, n = re.subn(rf'<w:style [^>]*w:styleId="{sid}"[^>]*>.*?</w:style>', lambda _: xml, s, count=1, flags=re.S)
                    if not n:
                        s = s.replace("</w:styles>", xml + "</w:styles>")
                data = s.encode()
            if item.filename == "word/document.xml":  # A4 page, 2 cm margins
                s = data.decode()
                s = re.sub(r"<w:pgSz[^>]*/>", '<w:pgSz w:w="11906" w:h="16838"/>', s)
                s = re.sub(r"<w:pgMar[^>]*/>", '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/>', s)
                data = s.encode()
            zout.writestr(item, data)
    default.unlink()


# ---------------------------------------------------------------- table post-processing
BORDERS = (f'<w:tblBorders><w:top w:val="single" w:sz="10" w:space="0" w:color="{NAVY}"/>'
           f'<w:bottom w:val="single" w:sz="10" w:space="0" w:color="{NAVY}"/>'
           f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{RULE}"/></w:tblBorders>')


# Numbered display equations: centre tab for the equation, right tab for its number
# (LibreOffice drops spacing commands inside OMML, so "\\qquad (1)" would sit against the equation).
EQ = (r"<w:p><w:pPr><w:pStyle w:val=\"BodyText\" /></w:pPr><m:oMathPara><m:oMathParaPr>.*?</m:oMathParaPr>"
      r"<m:oMath>(.*?)<m:r><m:rPr><m:nor /><m:sty m:val=\"p\" /></m:rPr><m:t>(\(\d+\))</m:t></m:r></m:oMath></m:oMathPara></w:p>")


def number_equation(m) -> str:
    return ('<w:p><w:pPr><w:pStyle w:val="BodyText"/><w:tabs><w:tab w:val="center" w:pos="4819"/>'
            '<w:tab w:val="right" w:pos="9638"/></w:tabs><w:jc w:val="left"/></w:pPr><w:r><w:tab/></w:r>'
            f'<m:oMath>{m.group(1)}</m:oMath><w:r><w:tab/><w:t>{m.group(2)}</w:t></w:r></w:p>')


def postprocess(docx: Path) -> None:
    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                s = data.decode()
                s = re.sub(EQ, number_equation, s, flags=re.S)
                s = re.sub(r"<w:tblW [^>]*/>", '<w:tblW w:w="5000" w:type="pct"/>', s)
                s = s.replace("<w:tblLook", BORDERS + "<w:tblLook")
                s = s.replace("</w:tbl>", '</w:tbl><w:p><w:pPr><w:spacing w:before="0" w:after="60"/></w:pPr></w:p>')

                def header(row):  # shade and bolden the first row of every table
                    row = re.sub(r"<w:tc>(?!<w:tcPr>)", "<w:tc><w:tcPr></w:tcPr>", row)
                    row = row.replace("</w:tcPr>", f'<w:shd w:val="clear" w:color="auto" w:fill="{HEAD_FILL}"/></w:tcPr>')
                    row = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr></w:rPr>", row)
                    return row.replace("</w:rPr>", "<w:b/><w:bCs/></w:rPr>")

                s = re.sub(r"(?<=<w:tblGrid>)(.*?</w:tblGrid>)(<w:tr>.*?</w:tr>)",
                           lambda m: m.group(1) + header(m.group(2)), s, flags=re.S)
                data = s.encode()
            zout.writestr(item, data)
    tmp.replace(docx)


# ---------------------------------------------------------------- build
def main() -> None:
    md = number_citations((HERE / SRC).read_text())
    words = len(re.sub(r"\(#[^)]*\)|\(https?://[^)]*\)|\{[^}]*\}|!\[|\$\$.*?\$\$", " ", md.split("# References")[0], flags=re.S).split())
    print(f"main text (title to conclusions/availability, incl. tables) ≈ {words} words")
    tmp = HERE / f".{SRC}.built.md"
    tmp.write_text(md)
    ref = HERE / ".reference_final.docx"
    make_reference_doc(ref)
    pypandoc.convert_file(str(tmp), "docx", outputfile=str(HERE / OUT),
                          extra_args=[f"--resource-path={HERE}", f"--reference-doc={ref}"])
    tmp.unlink()
    ref.unlink()
    postprocess(HERE / OUT)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(HERE / OUT)],
                   check=True, capture_output=True, timeout=300)
    print("built", OUT, "and", OUT.replace(".docx", ".pdf"))


if __name__ == "__main__":
    main()
