"""Extract key content from the research papers."""
import fitz  # pymupdf
import os

PAPERS_DIR = "PAPERS"

# Priority papers to read fully
PRIORITY_PAPERS = [
    ("Automatic_Modulation_Classification_A_Deep_Architecture_Survey.pdf", "DEEP ARCH SURVEY"),
    ("s41598-026-35558-7.pdf", "2026 AMC PAPER"),
    ("s44459-025-00013-y.pdf", "2025 AMC PAPER"),
    ("2503.04142v2.pdf", "2025 PREPRINT"),
    ("2009.07774v2.pdf", "2020 DATASET PAPER"),
    ("2601.15903v2.pdf", "2026 PREPRINT"),
    ("ilide.info-a-survey-of-modulation-classification-using-deep-learning-signal-representation--pr_f0aa7140ed37a6dd57c5e01cb6507bf5.pdf", "DL SURVEY"),
    ("entropy-22-01256.pdf", "ENTROPY AMC"),
    ("s13638-020-01730-4.pdf", "EURASIP AMC"),
    ("FINALVERSION.pdf", "FINAL VER"),
    ("FINALVERSION (1).pdf", "FINAL VER 2"),
    ("download.pdf", "DOWNLOAD"),
    ("paper_submitted.pdf", "PAPER SUBMITTED"),
]

def extract_paper(filename, label, max_pages=15):
    path = os.path.join(PAPERS_DIR, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {filename}")
        return
    
    print(f"\n{'='*80}")
    print(f"PAPER: {label}")
    print(f"FILE: {filename}")
    print('='*80)
    
    doc = fitz.open(path)
    total = len(doc)
    pages_to_read = min(max_pages, total)
    
    print(f"Total pages: {total}, reading first {pages_to_read}")
    print()
    
    for i in range(pages_to_read):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            print(f"--- PAGE {i+1} ---")
            print(text[:3000])
    
    doc.close()

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for fname, label in PRIORITY_PAPERS:
    extract_paper(fname, label, max_pages=8)
