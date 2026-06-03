from rank_bm25 import BM25Okapi

from src.store.pgvector_store import fetch_all_chunks


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseIndex:
    def __init__(self) -> None:
        self.chunks = fetch_all_chunks()
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked[:k]]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "retrieval augmented generation"
    print(f"Query: {query}\n")
    for i, hit in enumerate(SparseIndex().search(query), 1):
        print(f"{i}. [score {hit['score']:.2f}] {hit['title']}")
        print(f"   {hit['text'][:160]}...\n")
