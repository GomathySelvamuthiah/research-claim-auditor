"""
fetch_real_data.py

Fetches real-world retraction records from the CrossRef API and real paper
abstracts from the Semantic Scholar API. Both APIs are free and require no
authentication.

Usage:
    python scripts/fetch_real_data.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# Make project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CROSSREF_URL = "https://api.crossref.org/works"
_OPENALEX_URL = "https://api.openalex.org/works"
_MAILTO = "research@neu.edu"
_OA_HEADERS = {"User-Agent": f"ResearchClaimAuditor/1.0 (mailto:{_MAILTO})"}
_OA_INTER_REQUEST_SLEEP = 1   # seconds between queries (OpenAlex: 100 req/s)

queries = [
    "citation distortion academic papers misrepresentation",
    "neuroinflammation cognitive decline biomarkers",
    "gut microbiome depression anxiety mental health",
    "CRISPR gene editing cancer therapy",
    "metformin diabetes mortality cardiovascular",
    "retraction scientific misconduct reproducibility",
    "systematic review meta-analysis bias",
    "LLM large language model hallucination",
]


# ---------------------------------------------------------------------------
# Part 1 — CrossRef retraction records
# ---------------------------------------------------------------------------

def fetch_retractions(n: int = 200) -> list[dict]:
    """
    Fetch up to n retraction records from the CrossRef API across two pages
    (offset 0 and offset 100). Returns a flat list of dicts and saves to
    data/retraction_watch_real.csv.
    """
    records: list[dict] = []
    offsets = [0, 100]

    for offset in offsets:
        params = {
            "filter": "update-type:retraction",
            "rows": 100,
            "offset": offset,
            "mailto": _MAILTO,
        }
        try:
            resp = requests.get(_CROSSREF_URL, params=params, timeout=30)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except Exception as exc:
            print(f"   ⚠ CrossRef request failed (offset={offset}): {exc}")
            continue

        for i, item in enumerate(items):
            # Title
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""

            # First author
            authors = item.get("author", [])
            if authors:
                family = authors[0].get("family", "")
                given = authors[0].get("given", "")
                author = f"{family}, {given}".strip(", ")
            else:
                author = ""

            # Journal
            container = item.get("container-title", [""])
            journal = container[0] if container else ""

            # Year
            date_parts = (
                item.get("published", {})
                    .get("date-parts", [[""]])
            )
            year = date_parts[0][0] if date_parts and date_parts[0] else ""

            # Retraction date (deposited timestamp)
            retraction_date = (
                item.get("deposited", {})
                    .get("date-time", "")[:10]
            )

            records.append({
                "Record_ID": len(records) + 1,
                "Title": title,
                "Author": author,
                "Journal": journal,
                "Year": str(year),
                "DOI": item.get("DOI", ""),
                "Retraction_Date": retraction_date,
                "Reason": "See publisher notice",
            })

    # Trim to requested n
    records = records[:n]

    # Persist
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv("data/retraction_watch_real.csv", index=False)

    return records


# ---------------------------------------------------------------------------
# Part 2 — OpenAlex paper abstracts
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    OpenAlex stores abstracts as an inverted index {word: [positions]}.
    Reconstruct the original text by sorting word-position pairs.
    """
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def fetch_papers(
    queries: list[str],
    papers_per_query: int = 10,
) -> list[dict]:
    """
    Search OpenAlex for each query and collect papers that have abstracts.
    Deduplicates by DOI. Saves to data/real_papers.json.

    OpenAlex is fully open (no auth), rate limit 100 req/s, polite pool
    activated by supplying a mailto in the User-Agent header.
    """
    seen_dois: set[str] = set()
    papers: list[dict] = []

    for q_idx, query in enumerate(queries):
        try:
            resp = requests.get(
                _OPENALEX_URL,
                params={
                    "search": query,
                    "per-page": papers_per_query,
                    "filter": "has_abstract:true",
                },
                headers=_OA_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as exc:
            print(f"   ⚠ OpenAlex failed for '{query[:40]}': {exc}")
            results = []

        for item in results:
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
            if not abstract.strip():
                continue

            # DOI comes as full URL "https://doi.org/10.xxx/yyy" — strip prefix
            raw_doi = item.get("doi") or ""
            doi = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

            dedup_key = doi.lower() if doi else f"__no_doi_{len(papers)}"
            if doi and dedup_key in seen_dois:
                continue
            seen_dois.add(dedup_key)

            author_names = [
                a.get("author", {}).get("display_name", "")
                for a in (item.get("authorships") or [])
            ]

            papers.append({
                "doi": doi,
                "title": item.get("title", ""),
                "abstract": abstract,
                "year": item.get("publication_year"),
                "authors": author_names,
                "is_open_access": (item.get("open_access") or {}).get("is_oa", False),
            })

        # Polite delay
        if q_idx < len(queries) - 1:
            time.sleep(_OA_INTER_REQUEST_SLEEP)

    # Persist
    os.makedirs("data", exist_ok=True)
    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total": len(papers),
        "papers": papers,
    }
    with open("data/real_papers.json", "w") as f:
        json.dump(output, f, indent=2)

    return papers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fetching Real Data ===\n")

    print("1. CrossRef Retraction Database...")
    retractions = fetch_retractions(200)
    print(f"   ✅ {len(retractions)} retraction records saved to data/retraction_watch_real.csv")

    print("\n2. Semantic Scholar Papers...")
    papers = fetch_papers(queries)
    print(f"   ✅ {len(papers)} papers saved to data/real_papers.json")

    print("\n=== Sample Records ===")

    print("\nFirst retraction:")
    if retractions:
        print(json.dumps(retractions[0], indent=2))

    print("\nFirst paper:")
    if papers:
        p = papers[0]
        print(f"  Title: {p['title']}")
        print(f"  DOI: {p['doi']}")
        print(f"  Abstract: {p['abstract'][:150]}...")
