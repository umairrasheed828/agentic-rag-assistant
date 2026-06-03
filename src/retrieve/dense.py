from src.embed.embedder import embed_query
from src.store.pgvector_store import get_connection


def dense_search(query: str, k: int = 5) -> list[dict]:
    q = embed_query(query)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, title, text, embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q, q, k),
        ).fetchall()
    return [
        {"chunk_id": r[0], "title": r[1], "text": r[2], "distance": r[3]} for r in rows
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "How does retrieval augmented generation work?"
    print(f"Query: {query}\n")
    for i, hit in enumerate(dense_search(query), 1):
        print(f"{i}. [dist {hit['distance']:.3f}] {hit['title']}")
        print(f"   {hit['text'][:160]}...\n")
