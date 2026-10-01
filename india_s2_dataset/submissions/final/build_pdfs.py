#!/usr/bin/env python3
"""MAIN_REPORT.md and EVIDENCE_REPORT.md -> PDFs (headless Chrome).   python build_pdfs.py"""
import subprocess
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 17mm 16mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.3pt; line-height: 1.42; color: #111; }
h1 { font-size: 22pt; margin: 0 0 6pt; } h2 { font-size: 13.5pt; margin-top: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11.3pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 7pt 0; font-size: 9pt; } th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: left; }
th { background: #f1f1f1; } img { max-width: 100%; max-height: 200mm; display: block; margin: 6pt auto 2pt; }
p.caption { text-align: center; font-size: 8.8pt; color: #444; margin-top: 0; }
hr { border: 0; border-top: 1px solid #ccc; } h2, h3 { page-break-after: avoid; } img, table { page-break-inside: avoid; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #f3f6fb; border-left: 3px solid #2a78d6; font-size: 9.6pt; } blockquote p { margin: 2pt 0; }
a { color: #1a5fb4; text-decoration: none; word-break: break-all; }
.titlepage { page-break-after: always; text-align: center; padding-top: 55mm; }
.titlepage h1 { font-size: 26pt; } .titlepage h2 { border: 0; font-size: 15pt; color: #333; font-weight: normal; }
.titlepage table { margin: 22pt auto; font-size: 11.5pt; } .titlepage td, .titlepage th { border: 0; padding: 3pt 10pt; }
.titlepage th { background: none; }
"""
for name in ("MAIN_REPORT", "EVIDENCE_REPORT"):
    body = markdown.markdown((HERE / f"{name}.md").read_text(), extensions=["tables", "md_in_html"])
    body = body.replace("<p><em>Figure", '<p class="caption"><em>Figure')
    (HERE / f"{name}.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{name.replace('_', ' ').title()}</title><style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={HERE / f'{name}.pdf'}", (HERE / f"{name}.html").as_uri()], check=True, capture_output=True, timeout=180)
    print(name, subprocess.run(["pdfinfo", HERE / f"{name}.pdf"], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0], "pages")
