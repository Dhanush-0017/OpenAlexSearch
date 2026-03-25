import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# ── Step 1: Download a paper from arXiv ──────────────────────────────────────
ARXIV_PDF_URL = "https://arxiv.org/pdf/1706.03762"
PDF_PATH      = "test_paper.pdf"

print("Downloading paper from arXiv...")
response = requests.get(ARXIV_PDF_URL)
with open(PDF_PATH, "wb") as f:
    f.write(response.content)
print(f"Saved to {PDF_PATH}\n")

# ── Step 2: Load and Parse the PDF ───────────────────────────────────────────
print("Parsing PDF...")
reader    = SimpleDirectoryReader(input_files=[PDF_PATH])
documents = reader.load_data()
print(f"Total pages loaded: {len(documents)}\n")

# ── Step 3: Chunk the text into Nodes ────────────────────────────────────────
# In LlamaIndex, chunks are called "Nodes"
# chunk_size    = how many characters per node
# chunk_overlap = shared characters between consecutive nodes
splitter = SentenceSplitter(
    chunk_size    = 1024,
    chunk_overlap = 200,
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"Total chunks (nodes) created: {len(nodes)}\n")

# ── Step 4: Print first 3 nodes ──────────────────────────────────────────────
print("=" * 60)
print("SAMPLE CHUNKS (NODES)")
print("=" * 60)

for i, node in enumerate(nodes[:3], 1):
    print(f"\n--- Chunk {i} (page {node.metadata.get('page_label', '?')}) ---")
    print(node.text)
    print(f"\n[Length: {len(node.text)} characters]")
    print("-" * 60)

print(f"\nDone! {len(nodes)} total chunks from {len(documents)} pages.")

# ── Step 5: Send one chunk to Ollama and summarize ───────────────────────────
print("\n" + "=" * 60)
print("SENDING CHUNK TO OLLAMA")
print("=" * 60)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# pick node 1 (index 1) — contains the abstract
test_chunk = nodes[1].text

print(f"\nChunk being sent to Ollama:\n{test_chunk}\n")
print("Asking Ollama to summarize...")

payload = {
    "model" : OLLAMA_MODEL,
    "prompt": f"Summarize this section of a research paper in 2-3 sentences:\n\n{test_chunk}",
    "stream": False,
}

response = requests.post(OLLAMA_URL, json=payload, timeout=60)
summary  = response.json()["response"].strip()

print(f"\nOllama's Summary:\n{summary}")
