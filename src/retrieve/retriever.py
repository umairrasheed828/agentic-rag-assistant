from src.retrieve.hybrid import hybrid_search
from src.retrieve.rerank import rerank


def retrieve(query: str, k: int = 5) -> list[dict]:
    candidates = hybrid_search(query, k=20)  # dense + sparse + RRF shortlist
    return rerank(query, candidates, top_k=k)  # precise final ordering


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "How do you evaluate large language models?"
    print(f"Query: {query}\n")
    for i, hit in enumerate(retrieve(query), 1):
        print(f"{i}. [rerank {hit['rerank_score']:.2f}] {hit['title']}")
        print(f"   {hit['text'][:160]}...\n")
