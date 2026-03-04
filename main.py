import requests
from pyalex import Works

# ─────────────────────────────
# SETTINGS — change these anytime
# ─────────────────────────────
KEYWORD   = "LLM"
YEAR_FROM = 2024
MAX_PAPERS = 5          # start small for testing

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"


# ─────────────────────────────
# STEP 1: Get papers from OpenAlex
# ─────────────────────────────
def get_papers():
    print(f"\n🔍 Searching OpenAlex for: '{KEYWORD}'")
    
    results = (
        Works()
        .search(KEYWORD)
        .filter(publication_year=f">{YEAR_FROM - 1}")
        .filter(type="article")
        .get()
    )
    
    papers = []
    for paper in results:
        if paper["abstract"] is None:
            continue
        papers.append({
            "title"   : paper["title"],
            "abstract": paper["abstract"],
            "year"    : paper["publication_year"]
        })
        if len(papers) >= MAX_PAPERS:
            break
    
    print(f"✅ Found {len(papers)} papers with abstracts\n")
    return papers


# ─────────────────────────────
# STEP 2: Classify with Ollama
# ─────────────────────────────
def classify(title, abstract):
    prompt = f"""You are a research paper classifier.

Read this paper title and abstract and classify it as ONLY one of:
- "efficiency"  → improving speed, reducing compute, memory, cost
- "scaling"     → scaling laws, larger models, more data
- "both"        → covers efficiency AND scaling
- "neither"     → something else

Reply with ONE word only: efficiency, scaling, both, or neither.

Title: {title}
Abstract: {abstract}
"""
    
    response = requests.post(
        OLLAMA_URL,
        json={
            "model" : OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    
    result = response.json()["response"].strip().lower()
    return result


# ─────────────────────────────
# STEP 3: Run everything
# ─────────────────────────────
def main():
    print("=" * 50)
    print("  OpenAlex + Ollama Classifier")
    print("=" * 50)
    
    # Get papers
    papers = get_papers()
    
    # Classify each one
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {paper['title'][:60]}...")
        classification = classify(paper["title"], paper["abstract"])
        print(f"     Result: {classification.upper()}\n")


if __name__ == "__main__":
    main()
