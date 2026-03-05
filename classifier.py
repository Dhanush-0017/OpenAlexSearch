import requests
import csv
import json
from pyalex import Works

# ─────────────────────────────────────────────
# SETTINGS — only change these if needed
# ─────────────────────────────────────────────

YEAR_FROM  = 2023
LANGUAGE   = "en"
MAX_PAPERS = 10

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

OUTPUT_CSV  = "results.csv"
OUTPUT_JSON = "results_full.json"


# ─────────────────────────────────────────────
# STEP 1: ASK USER FOR KEYWORD
# ─────────────────────────────────────────────

def get_keyword():
    print("=" * 50)
    print("   OpenAlex Paper Search")
    print("=" * 50)
    
    # Ask user to type a keyword
    keyword = input("\nEnter keyword to search: ").strip()
    
    # Don't allow empty keyword
    if not keyword:
        print("Keyword cannot be empty!")
        return get_keyword()  # ask again
    
    return keyword


# ─────────────────────────────────────────────
# STEP 2: GET PAPERS FROM OPENALEX
# ─────────────────────────────────────────────

def get_papers(keyword):
    print(f"\n Searching OpenAlex for: '{keyword}'")
    print(f" Filters: year >= {YEAR_FROM} | language = {LANGUAGE} | type = article\n")

    try:
        results = (
            Works()
            .search(keyword)
            .filter(publication_year=f">{YEAR_FROM - 1}")
            .filter(type="article")
            .filter(language=LANGUAGE)
            .get()
        )

    except Exception as e:
        print(f" Error connecting to OpenAlex: {e}")
        return []

    # Collect papers that have both title and abstract
    papers = []
    for paper in results:

        if paper["abstract"] is None:
            continue

        if paper["title"] is None:
            continue

        papers.append({
            "title"   : paper["title"],
            "abstract": paper["abstract"],
            "year"    : paper["publication_year"],
            "doi"     : paper.get("doi", "N/A"),
        })

        if len(papers) >= MAX_PAPERS:
            break

    print(f" Found {len(papers)} papers with abstracts")
    return papers


# ─────────────────────────────────────────────
# STEP 3: CLASSIFY PAPER USING OLLAMA
# ─────────────────────────────────────────────

def classify_paper(title, abstract):
    prompt = f"""You are a research paper classifier. Be very careful and precise.

DEFINITIONS:
- "efficiency"  -> the paper focuses on making models FASTER, SMALLER, or CHEAPER
                   Keywords: quantization, pruning, compression, edge deployment,
                   inference speed, memory reduction, energy consumption, lightweight
                   
- "scaling"     -> the paper focuses on making models BIGGER or using MORE DATA
                   Keywords: scaling laws, larger models, more parameters, more data,
                   training compute, billion parameters

- "both"        -> the paper CLEARLY covers both efficiency AND scaling equally

- "neither"     -> the paper is about something else like applications,
                   benchmarks, datasets, or social impact

IMPORTANT: If the abstract mentions reducing cost, memory, or compute -> likely "efficiency"
           If the abstract mentions bigger models or more data -> likely "scaling"

You must respond in this EXACT JSON format and nothing else:
{{
  "classification": "efficiency" or "scaling" or "both" or "neither",
  "reason": "one sentence explaining why"
}}

No extra text. No markdown. Just the JSON object.

Title: {title}

Abstract: {abstract}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model" : OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        raw = response.json()["response"].strip()

        # Clean up in case LLM adds markdown code fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Parse the JSON response
        result = json.loads(raw)

        # Validate classification label
        valid_labels = ["efficiency", "scaling", "both", "neither"]
        if result["classification"] not in valid_labels:
            result["classification"] = "unclear"

        return result

    except requests.exceptions.ConnectionError:
        print("  Could not connect to Ollama. Is it running?")
        return {"classification": "error", "reason": "Could not connect to Ollama"}

    except json.JSONDecodeError:
        print(f"  LLM gave unexpected response: {raw[:100]}")
        return {"classification": "unclear", "reason": "LLM response could not be parsed"}

    except Exception as e:
        print(f"  Ollama error: {e}")
        return {"classification": "error", "reason": str(e)}
# ─────────────────────────────────────────────
# STEP 4: SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(results):

    # --- Save clean summary to CSV (for Excel) ---
    columns = ["title", "year", "doi", "classification", "reason"]

    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "title"         : row["title"],
                    "year"          : row["year"],
                    "doi"           : row["doi"],
                    "classification": row["classification"],
                    "reason"        : row["reason"]
                })
        print(f"\n Summary saved to: {OUTPUT_CSV} (open in Excel)")

    except Exception as e:
        print(f" Could not save CSV: {e}")

    # --- Save full data with abstracts to JSON ---
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f" Full data saved to: {OUTPUT_JSON} (includes abstracts)")

    except Exception as e:
        print(f" Could not save JSON: {e}")


# ─────────────────────────────────────────────
# STEP 5: MAIN — TIE EVERYTHING TOGETHER
# ─────────────────────────────────────────────

def main():

    # Ask user for keyword
    keyword = get_keyword()

    # Get papers from OpenAlex
    papers = get_papers(keyword)

    # Stop if no papers found
    if not papers:
        print(" No papers found. Try a different keyword.")
        return

    # Process each paper
    results = []
    for i, paper in enumerate(papers, 1):

        # Print paper details
        print(f"\n{'=' * 50}")
        print(f"[{i}/{len(papers)}] {paper['title']}")
        print(f"Year: {paper['year']}  |  DOI: {paper['doi']}")
        print(f"\nAbstract:\n{paper['abstract']}")

        # Classify the paper
        print(f"\n Classifying...")
        result = classify_paper(paper["title"], paper["abstract"])
        
        print(f" Result: {result['classification'].upper()}")
        print(f" Reason: {result['reason']}")

        # Store result
        paper["classification"] = result["classification"]
        paper["reason"]         = result["reason"]
        results.append(paper)

    # Print summary
    print(f"\n{'=' * 50}")
    print(" SUMMARY")
    print(f"{'=' * 50}")

    from collections import Counter
    counts = Counter(r["classification"] for r in results)
    for label, count in counts.most_common():
        print(f"  {label.capitalize():12s}: {count} papers")

    # Save both files
    save_results(results)


if __name__ == "__main__":
    main()


