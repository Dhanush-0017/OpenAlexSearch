import csv
import json
import os
from collections import Counter
from search import get_keyword, get_papers, DEFAULT_YEAR_FROM, DEFAULT_MAX_PAPERS
from classifier import classify_paper

RESULTS_DIR = "results"

ollama_url   = "http://localhost:11434/api/generate"
#ollama_url   = "http://10.230.100.240:17020/api/generate"

ollama_model = "llama3.2"


def get_settings():
    #ask user for year and max papers - press Enter to keep defaults
    print("\nSettings (press Enter to keep default):")

    year_input = input(f"  From year [{DEFAULT_YEAR_FROM}]: ").strip()
    if year_input.isdigit() and len(year_input) == 4:
        
        year_from = int(year_input)
    else:
        year_from = DEFAULT_YEAR_FROM

    count_input = input(f"  Max papers [{DEFAULT_MAX_PAPERS}]: ").strip()
    
    if count_input.isdigit() and 1 <= int(count_input) <= 200:
        max_papers = int(count_input)
    else:
        max_papers = DEFAULT_MAX_PAPERS

    print(f"  -> Searching from {year_from}, up to {max_papers} papers\n")
    return year_from, max_papers


def save_results(results):
    #create results folder if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_csv  = os.path.join(RESULTS_DIR, "results.csv")
    output_json = os.path.join(RESULTS_DIR, "results.json")

    #save a clean summary to csv for excel
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            columns = ["title", "authors", "year", "journal", "doi", "cited_by", "classification", "accuracy", "reason", "concepts", "keywords", "bibtex"]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "title"         : row["title"],
                    "authors"       : row.get("authors", "N/A"),
                    "year"          : row["year"],
                    "journal"       : row.get("journal", "N/A"),
                    "doi"           : row["doi"],
                    "cited_by"      : row.get("cited_by", 0),
                    "classification": row["classification"],
                    "accuracy"      : str(row.get("accuracy", 0)) + "%",
                    "reason"        : row["reason"],
                    "concepts"      : row.get("concepts", "N/A"),
                    "keywords"      : row.get("keywords", "N/A"),
                    "bibtex"        : row.get("bibtex", ""),
                })
        print(f"\nSummary saved to {output_csv}")

    except PermissionError:
        #this happens when the csv file is open in excel
        print(f"Please close {output_csv} in Excel and try again.")

    except Exception as e:
        print(f"Could not save CSV: {e}")

    #save full data including abstracts to json
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full data saved to {output_json}")

    except Exception as e:
        print(f"Could not save JSON: {e}")


def main():
    #keep running until user wants to stop
    while True:
        print("=" * 50)
        print("   OpenAlex Paper Search")
        print("=" * 50)

        #step 1 - ask user for a keyword
        keyword = get_keyword()

        #step 2 - ask user for settings (year, max papers)
        year_from, max_papers = get_settings()

        #step 3 - get papers from openalex using user settings
        papers = get_papers(keyword, max_papers=max_papers, year_from=year_from)

        #stop if no papers were found
        if not papers:
            print("No papers found. Please try a different keyword.")
        else:
            #step 4 - classify each paper using ollama
            results = []
            ollama_is_down = False

            for i, paper in enumerate(papers, 1):

                #print paper details (truncate abstract to keep terminal clean)
                print(f"\n{'-' * 50}")
                print(f"[{i}/{len(papers)}] {paper['title']}")
                print(f"Year : {paper['year']}  |  DOI: {paper['doi']}")
                print(f"\nAbstract:\n{paper['abstract'][:300]}...")

                #send to ollama and get classification
                print(f"\nClassifying...")
                result = classify_paper(
                    paper["title"],
                    paper["abstract"],
                    ollama_url,
                    ollama_model,
                    journal  = paper.get("journal", "N/A"),
                    concepts = paper.get("concepts", "N/A"),
                    keywords = paper.get("keywords", "N/A"),
                )

                #if ollama is not running stop the loop
                if result["classification"] == "error" and "could not connect" in result["reason"]:
                    print("\nOllama is not running. Please start Ollama and try again.")
                    ollama_is_down = True
                    break

                print(f"Result : {result['classification'].upper()}  [accuracy: {result.get('accuracy', 0)}%]")
                print(f"Reason : {result['reason']}")

                #add classification and reason to paper
                paper["classification"] = result["classification"]
                paper["accuracy"]       = result.get("accuracy", 0)
                paper["reason"]         = result["reason"]
                results.append(paper)

            #step 5 - show summary and save only if we have results
            if results and not ollama_is_down:

                print(f"\n{'=' * 50}")
                print("SUMMARY")
                print(f"{'=' * 50}")

                counts = Counter(r["classification"] for r in results)
                for label, count in counts.most_common():
                    print(f"  {label.capitalize():12s}: {count} papers")

                #step 6 - save results named after the keyword
                save_results(results)

                print(f"\nDone! Processed {len(results)} papers.")

        #ask if user wants to search again
        print("\n" + "=" * 50)
        again = input("Do you want to search again? (yes/no): ").strip().lower()

        if again != "yes":
            print("\nGoodbye!")
            break



if __name__ == "__main__":
    main()
