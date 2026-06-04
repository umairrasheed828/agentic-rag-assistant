import json
from pathlib import Path

JUDGE = Path("eval/judgments.jsonl")
HUMAN = Path("eval/human_labels.jsonl")
AXES = ("faithfulness", "relevance", "correctness")


def load(path: Path) -> dict:
    return {
        json.loads(line)["question"]: json.loads(line)
        for line in path.open(encoding="utf-8")
    }


def main() -> None:
    judge, human = load(JUDGE), load(HUMAN)
    common = [q for q in judge if q in human]
    print(f"Comparing {len(common)} judged answers against human labels.\n")

    for axis in AXES:
        diffs = [abs(judge[q][axis] - human[q][axis]) for q in common]
        mae = sum(diffs) / len(diffs)
        exact = sum(1 for d in diffs if d == 0)
        print(f"{axis:13} MAE={mae:.2f}  exact-agreement={exact}/{len(common)}")

    print("\n--- biggest disagreements ---")
    scored = [(sum(abs(judge[q][a] - human[q][a]) for a in AXES), q) for q in common]
    for total, q in sorted(scored, reverse=True)[:3]:
        if total == 0:
            continue
        print(f"[Δ{total}] {q[:55]}")
        for a in AXES:
            if judge[q][a] != human[q][a]:
                print(f"    {a}: judge={judge[q][a]} human={human[q][a]}")


if __name__ == "__main__":
    main()
