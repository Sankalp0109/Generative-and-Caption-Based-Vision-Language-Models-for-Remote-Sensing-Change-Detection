#!/usr/bin/env python3
"""REPORT.md -> report.html -> REPORT.pdf (headless Chrome).   python submissions/build_pdf.py"""
import subprocess
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
body = markdown.markdown((HERE / "REPORT.md").read_text(), extensions=["tables"])
body = body.replace("<p><em>Figure", '<p class="caption"><em>Figure')
css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 180mm; margin: auto; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 13.5pt; margin-top: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; } th, td { border: 1px solid #bbb; padding: 3pt 7pt; text-align: left; }
th { background: #f1f1f1; } img { max-width: 100%; max-height: 205mm; display: block; margin: 8pt auto 2pt; }
p.caption { text-align: center; font-size: 9pt; color: #444; margin-top: 0; }
hr { border: 0; border-top: 1px solid #ccc; } h2, h3 { page-break-after: avoid; } img, table { page-break-inside: avoid; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Remote Sensing Image Change Caption Generation</title><style>{css}</style></head><body>{body}</body></html>"
(HERE / "report.html").write_text(html)
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={HERE / 'REPORT.pdf'}", (HERE / "report.html").as_uri()], check=True, capture_output=True, timeout=120)
print("wrote", HERE / "REPORT.pdf")
