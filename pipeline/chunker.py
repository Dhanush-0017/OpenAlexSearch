import os
import json
import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


# ── Constants ─────────────────────────────────────────────────────────────────
PAPERS_DIR = "papers"   # folder where PDFs are saved
OUTPUT_DIR = "output"   # folder where JSON results are saved

CHUNK_SIZE    = 1024  # max characters per chunk
CHUNK_OVERLAP = 200   # shared characters between chunks


# ── Step 1: Download PDF ──────────────────────────────────────────────────────
def download_pdf(paper_id, pdf_url):
    """
    Downloads a PDF from a given URL and saves it to the papers/ folder.
    Returns the local file path of the saved PDF.
    """
    os.makedirs(PAPERS_DIR, exist_ok=True)

    pdf_path = os.path.join(PAPERS_DIR, f"{paper_id}.pdf")

    print(f"Downloading paper {paper_id}...")
    response = requests.get(pdf_url, timeout=30)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    print(f"Saved to {pdf_path}")
    return pdf_path


# ── Step 2: Parse and Chunk PDF ───────────────────────────────────────────────
def chunk_pdf(pdf_path):
    """
    Reads a PDF file and splits it into chunks (nodes) using LlamaIndex.
    Returns a list of nodes where each node has text and metadata.
    """
    print(f"Parsing and chunking {pdf_path}...")

    # load the PDF — SimpleDirectoryReader reads the file page by page
    reader    = SimpleDirectoryReader(input_files=[pdf_path])
    documents = reader.load_data()

    # chunk the text into nodes using SentenceSplitter
    # SentenceSplitter is smarter than character splitting —
    # it tries to keep complete sentences together
    splitter = SentenceSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks from {len(documents)} pages")

    return nodes


# ── Step 3: Save to JSON ──────────────────────────────────────────────────────
def save_chunks(paper_id, title, source, nodes):
    """
    Saves all chunks for a paper into a single JSON file in the output/ folder.
    Each chunk includes its id, page number, and text.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # build the list of chunks from nodes
    chunks = []
    for i, node in enumerate(nodes, 1):
        chunks.append({
            "chunk_id" : i,
            "page"     : node.metadata.get("page_label", "?"),
            "text"     : node.text,
        })

    # build the full paper result
    result = {
        "paper_id"    : paper_id,
        "title"       : title,
        "source"      : source,
        "total_pages" : len(set(c["page"] for c in chunks)),
        "total_chunks": len(chunks),
        "chunks"      : chunks,
    }

    # save to output/paper_id.json
    output_path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path}")
    return output_path


# ── Main function — runs the full chunking pipeline ───────────────────────────
def process_paper(paper_id, title, source, pdf_url):
    """
    Full pipeline for one paper:
    1. Download PDF
    2. Parse and chunk
    3. Save to JSON
    """
    print(f"\n{'='*60}")
    print(f"Processing: {title}")
    print(f"{'='*60}")

    pdf_path    = download_pdf(paper_id, pdf_url)
    nodes       = chunk_pdf(pdf_path)
    output_path = save_chunks(paper_id, title, source, nodes)

    print(f"Done! Output saved to {output_path}")
    return output_path


# ── Run directly to test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    process_paper(
        paper_id = "1706.03762",
        title    = "Attention Is All You Need",
        source   = "arxiv",
        pdf_url  = "https://arxiv.org/pdf/1706.03762",
    )
