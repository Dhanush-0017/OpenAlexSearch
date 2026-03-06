# search.py
# this file connects to OpenAlex and gets papers with abstracts

import os
import pyalex
from pyalex import Works
import requests

# load api key from .env file (keeps it out of the code)
try:
    from dotenv import load_dotenv
    load_dotenv()
    pyalex.config.api_key = os.environ.get("OPENALEX_API_KEY", "")
except ImportError:
    # if python-dotenv is not installed, fall back to hardcoded key
    print("Tip: run 'pip install python-dotenv' to load API key from .env file")
    pyalex.config.api_key = "wdxmWbVkwxbgPOfwdwxAgi"

# default settings - used if user does not change them
DEFAULT_YEAR_FROM  = 2023
DEFAULT_LANGUAGE   = "en"
DEFAULT_MAX_PAPERS = 10


def get_keyword():
    # ask the user to type a keyword
    keyword = input("Enter keyword to search: ").strip()

    # if user typed nothing, ask again
    if not keyword:
        print("Please enter a keyword!")
        return get_keyword()

    # if keyword is too short, ask again
    if len(keyword) < 2:
        print("Keyword is too short. Please try again.")
        return get_keyword()

    return keyword


def get_papers(keyword, max_papers=DEFAULT_MAX_PAPERS, year_from=DEFAULT_YEAR_FROM):
    # search OpenAlex for papers matching the keyword
    print(f"\nSearching for '{keyword}' on OpenAlex...")
    print(f"Filters: year >= {year_from}  |  language = {DEFAULT_LANGUAGE}  |  max papers = {max_papers}")

    papers = []

    try:
        # use paginate() instead of get() so we fetch more pages if needed
        # this means if the first 25 results lack abstracts we keep going
        pages = (
            Works()
            .search(keyword)
            .filter(publication_year=f">{year_from - 1}")
            .filter(type="article")
            .filter(language=DEFAULT_LANGUAGE)
            .paginate(per_page=25)
        )

        for page in pages:
            for paper in page:

                # skip if no title
                title = paper["title"]
                if not title:
                    continue

                # use paper["abstract"] not paper.get("abstract")
                # pyalex reconstructs abstract from abstract_inverted_index via __getitem__
                abstract = paper["abstract"]
                if not abstract:
                    continue

                # save the paper details we need
                papers.append({
                    "title"   : title,
                    "abstract": abstract,
                    "year"    : paper["publication_year"],
                    "doi"     : paper.get("doi", "N/A"),
                })

                # stop when we have enough papers
                if len(papers) >= max_papers:
                    break

            # also break out of the pages loop when we have enough
            if len(papers) >= max_papers:
                break

    except requests.exceptions.ConnectionError:
        # this happens when there is no internet connection
        print("No internet connection. Please check your connection and try again.")
        return []

    except requests.exceptions.Timeout:
        # this happens when OpenAlex takes too long to respond
        print("OpenAlex is taking too long to respond. Please try again.")
        return []

    except Exception as e:
        # this catches any other unexpected error
        print(f"Something went wrong with OpenAlex: {type(e).__name__}: {e}")
        return []

    if not papers:
        return []

    print(f"Found {len(papers)} papers with abstracts.")
    return papers
