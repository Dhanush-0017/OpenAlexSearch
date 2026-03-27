import os
import re
import json
import logging
import warnings
import fitz  # PyMuPDF

warnings.filterwarnings("ignore")
for _log in ["httpx", "httpcore", "huggingface_hub", "llama_index", "transformers", "sentence_transformers"]:
    logging.getLogger(_log).setLevel(logging.ERROR)

OUTPUT_DIR = "output"


# ── Step 1: Read PDF ──────────────────────────────────────────────────────────
def read_pdf(pdf_path):
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    return full_text, doc


# ── Step 2: Extract Title using font size ─────────────────────────────────────
def extract_title(doc, metadata):
    # Try PDF metadata first (skip if it looks like an arxiv ID)
    title = (metadata.get("title") or "").strip()
    if title and len(title) > 5 and not title.lower().startswith("arxiv"):
        return title

    # Find the text span with the largest font size on page 1,
    # skipping arxiv stamps, dates, URLs, and other non-title text
    page      = doc[0]
    blocks    = page.get_text("dict")["blocks"]
    best_size = 0
    best_text = ""

    skip_patterns = re.compile(
        r'(arxiv|preprint|\[cs\.|http|©|copyright|submission|proceedings|conference|\d{4}-\d{2}-\d{2})',
        re.IGNORECASE
    )

    # Pass 1: find the largest font size used in a valid title line
    all_lines = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = " ".join(s["text"] for s in line.get("spans", [])).strip()
            if not line_text or skip_patterns.search(line_text):
                continue
            line_max_size = max((s["size"] for s in line.get("spans", [])), default=0)
            if 8 < len(line_text) < 200:
                all_lines.append((line_max_size, line_text))
                if line_max_size > best_size:
                    best_size = line_max_size

    # Pass 2: collect ALL lines at the largest font size (titles often span multiple lines)
    title_lines = [
        text for size, text in all_lines
        if size >= best_size - 0.5
    ]
    title = " ".join(title_lines) if title_lines else "Unknown Title"
    return fix_spaced_title(title)


def fix_spaced_title(title):
    """Fix PDF small-caps font artifacts that split words with spaces.

    Handles two patterns:
      'F LASH A TTENTION' → 'FLASH ATTENTION'  (single letter + rest of word)
      'QL O RA'           → 'QLORA'             (all short tokens before colon)
    """
    # Fix single uppercase letter split from the rest: "F LASH" → "FLASH"
    title = re.sub(r'\b([A-Z])\s+([A-Z]{2,})\b', r'\1\2', title)

    # Fix all-short tokens before colon: "QL O RA: ..." → "QLORA: ..."
    parts  = title.split(":", 1)
    prefix = parts[0].strip()
    words  = prefix.split()
    if len(words) >= 2 and all(len(w) <= 3 for w in words):
        parts[0] = "".join(words)
        return ": ".join(parts)

    return title


# ── Step 3: Extract Abstract, Introduction, Conclusion ───────────────────────
def extract_sections(full_text):
    sections = {"abstract": "", "introduction": "", "conclusion": ""}

    abstract_match = re.search(
        r'(?i)\babstract\b[:\s]*\n(.*?)(?=\n\s*(?:1[\.\s]|introduction\b|\Z))',
        full_text, re.DOTALL
    )
    if abstract_match:
        sections["abstract"] = abstract_match.group(1).strip()

    intro_match = re.search(
        r'(?i)(?:1[\.\s]+)?introduction\s*\n(.*?)(?=\n\s*(?:\d+[\.\s]+\w|related work|background|methodology|methods)\b)',
        full_text, re.DOTALL
    )
    if intro_match:
        sections["introduction"] = intro_match.group(1).strip()

    conclusion_match = re.search(
        r'(?i)(?:\d+[\.\s]+)?conclusions?\s*\n(.*?)(?=\n\s*(?:references|bibliography|acknowledgem)\b|\Z)',
        full_text, re.DOTALL
    )
    if conclusion_match:
        sections["conclusion"] = conclusion_match.group(1).strip()

    return sections


# ── Step 4: Section-based chunking (Abstract, Introduction, Conclusion) ───────
def chunk_by_section(sections):
    chunks = []
    for i, (section_name, text) in enumerate(sections.items()):
        text = text.strip()
        if text:
            chunks.append({
                "chunk_index": i,
                "section"    : section_name,
                "text"       : text,
                "char_count" : len(text),
            })
    return chunks


# ── Step 5: Save to JSON ──────────────────────────────────────────────────────
def save_chunks(paper_id, title, pdf_path, sections, chunks):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {
        "paper_id"          : paper_id,
        "title"             : title,
        "pdf_path"          : pdf_path,
        "sections_extracted": list(sections.keys()),
        "num_chunks"        : len(chunks),
        "chunks"            : chunks,
        "section_summaries" : [],
        "summary"           : "",
        "classification"    : {},
    }
    with open(os.path.join(OUTPUT_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def process_pdf(pdf_path, paper_id=None, title=None):
    if not paper_id:
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    full_text, doc = read_pdf(pdf_path)

    if not title:
        title = extract_title(doc, doc.metadata)

    sections = extract_sections(full_text)
    chunks   = chunk_by_section(sections)

    save_chunks(paper_id, title, pdf_path, sections, chunks)
    return chunks, title
