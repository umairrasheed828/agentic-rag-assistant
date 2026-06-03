from src.retrieve.dense import dense_search
from src.retrieve.sparse import SparseIndex
from src.store.pgvector_store import fetch_all_chunks


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists of chunk_ids into one fused ranking."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    dense_ids = [hit["chunk_id"] for hit in dense_search(query, k=20)]
    sparse_ids = [hit["chunk_id"] for hit in SparseIndex().search(query, k=20)]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    by_id = {c["chunk_id"]: c for c in fetch_all_chunks()}
    return [by_id[doc_id] for doc_id, _ in fused[:k] if doc_id in by_id]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "How do you evaluate large language models?"
    print(f"Query: {query}\n")
    for i, hit in enumerate(hybrid_search(query), 1):
        print(f"{i}. {hit['title']}")
        print(f"   {hit['text'][:160]}...\n")
