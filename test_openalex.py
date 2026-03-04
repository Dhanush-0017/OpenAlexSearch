from pyalex import Works

# Search with filters
results = (
    Works()
    .search("LLM")
    .filter(publication_year=">2023")
    .filter(type="article")
    .get()
)

# Print first 3 papers
for paper in results[:3]:
    
    # Skip if no abstract
    if paper["abstract"] is None:
        print("No abstract, skipping...")
        print("---")
        continue
    
    print("Title:", paper["title"])
    print("Year:", paper["publication_year"])
    print("Abstract:", paper["abstract"][:200])
    print("---")
