import json
from pathlib import Path

from src.agent.llm import get_llm
from src.agent.rag import SYSTEM_PROMPT
from src.retrieve.retriever import retrieve

QA = Path("eval/qa_set.jsonl")
OUT = Path("eval/eval_run.jsonl")


def run() -> int:
    llm = get_llm()
    cases = [json.loads(line) for line in QA.open(encoding="utf-8")]
    with OUT.open("w", encoding="utf-8") as f:
        for case in cases:
            chunks = retrieve(case["question"], k=5)
            contexts = [c["text"] for c in chunks]
            context_str = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
            messages = [
                ("system", SYSTEM_PROMPT.format(context=context_str)),
                ("human", case["question"]),
            ]
            answer = str(llm.invoke(messages).content)
            f.write(
                json.dumps(
                    {
                        "question": case["question"],
                        "answer": answer,
                        "contexts": contexts,
                        "reference": case["reference_answer"],
                    }
                )
                + "\n"
            )
    return len(cases)


if __name__ == "__main__":
    n = run()
    print(f"Wrote {n} eval records to {OUT}")
