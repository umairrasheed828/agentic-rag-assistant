import json
from pathlib import Path

from src.retrieve.retriever import retrieve

GOLD = Path("eval/gold_set.jsonl")


def evaluate(k: int = 5) -> None:
    cases = [json.loads(line) for line in GOLD.open(encoding="utf-8")]
    hits = 0
    reciprocal_ranks = []

    for case in cases:
        results = retrieve(case["query"], k=k)
        target = case["relevant_title_contains"].lower()
        rank = next(
            (i for i, r in enumerate(results, 1) if target in r["title"].lower()),
            None,
        )
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        print(f"[{'hit@' + str(rank) if rank else 'MISS'}] {case['query']}")

    n = len(cases)
    print(f"\nRecall@{k}: {hits}/{n} = {hits / n:.2f}")
    print(f"MRR:       {sum(reciprocal_ranks) / n:.3f}")


if __name__ == "__main__":
    evaluate()
