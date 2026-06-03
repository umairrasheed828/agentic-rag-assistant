import json
from pathlib import Path

import arxiv

OUTPUT = Path("data/papers.jsonl")


def download_papers(query: str, max_results: int = 100) -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=5.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    count = 0
    with OUTPUT.open("w", encoding="utf-8") as f:
        for result in client.results(search):
            record = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title.strip(),
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    n = download_papers(
        query="retrieval augmented generation OR LLM agents OR LLM evaluation",
        max_results=50,
    )
    print(f"Saved {n} papers to {OUTPUT}")
