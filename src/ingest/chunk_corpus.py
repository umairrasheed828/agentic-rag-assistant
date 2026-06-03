import json
from pathlib import Path

from src.chunk import chunk_text

INPUT = Path("data/papers.jsonl")
OUTPUT = Path("data/chunks.jsonl")


def build_chunks(chunk_size: int = 120, overlap: int = 20) -> int:
    count = 0
    with (
        INPUT.open(encoding="utf-8") as fin,
        OUTPUT.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            paper = json.loads(line)
            text = f"{paper['title']}. {paper['abstract']}"
            for i, chunk in enumerate(
                chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            ):
                record = {
                    "chunk_id": f"{paper['arxiv_id']}_{i}",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "text": chunk,
                }
                fout.write(json.dumps(record) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    n = build_chunks()
    print(f"Wrote {n} chunks to {OUTPUT}")
