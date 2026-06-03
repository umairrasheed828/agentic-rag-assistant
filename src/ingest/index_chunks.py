import json
from pathlib import Path

from src.embed.embedder import embed_passages
from src.store.pgvector_store import count_chunks, insert_chunks

CHUNKS = Path("data/chunks.jsonl")


def index_chunks(batch_size: int = 32) -> int:
    records = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        embeddings = embed_passages([r["text"] for r in batch])
        rows = [
            {
                "chunk_id": r["chunk_id"],
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "text": r["text"],
                "embedding": emb,
            }
            for r, emb in zip(batch, embeddings)
        ]
        insert_chunks(rows)
        total += len(rows)
        print(f"  indexed {total}/{len(records)}")
    return total


if __name__ == "__main__":
    n = index_chunks()
    print(f"Done. {n} chunks stored. Total rows in DB: {count_chunks()}")
