import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.agent.llm import get_llm

EVAL_RUN = Path("eval/eval_run.jsonl")
OUT = Path("eval/judgments.jsonl")


class Judgment(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5, description="Is every claim supported by the CONTEXT?"
    )
    relevance: int = Field(
        ge=1, le=5, description="Does the ANSWER address the QUESTION?"
    )
    correctness: int = Field(
        ge=1, le=5, description="Does the ANSWER agree with the REFERENCE?"
    )
    rationale: str = Field(description="One sentence explaining the scores.")


JUDGE_PROMPT = """You are a strict evaluator of a RAG system's answer. \
Score each axis from 1 (poor) to 5 (excellent):
- faithfulness: is every claim supported by the CONTEXT? Be conservative — give 5 ONLY if EVERY claim appears explicitly in the context. If the answer adds plausible-but-unstated detail, score 3 or lower.
- relevance: does the ANSWER actually address the QUESTION?
- correctness: does the ANSWER agree with the REFERENCE? Give 5 ONLY if it fully matches with no extra unsupported claims; deduct for additions the reference doesn't make.
Give a one-sentence rationale.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

REFERENCE:
{reference}"""


def judge() -> None:
    llm = get_llm().with_structured_output(Judgment)
    records = [json.loads(line) for line in EVAL_RUN.open(encoding="utf-8")]

    results = []
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            prompt = JUDGE_PROMPT.format(
                question=r["question"],
                context="\n\n".join(r["contexts"]),
                answer=r["answer"],
                reference=r["reference"],
            )
            j: Judgment = llm.invoke(prompt)  # typed Judgment, not raw text
            row = {"question": r["question"], **j.model_dump()}
            results.append(row)
            f.write(json.dumps(row) + "\n")
            print(
                f"[F{j.faithfulness} R{j.relevance} C{j.correctness}] {r['question'][:50]}"
            )

    n = len(results)
    print("\n--- averages ---")
    for axis in ("faithfulness", "relevance", "correctness"):
        print(f"{axis}: {sum(x[axis] for x in results) / n:.2f}")


if __name__ == "__main__":
    judge()
