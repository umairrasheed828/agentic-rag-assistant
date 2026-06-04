import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()  # so LANGSMITH_API_KEY is available

QA = Path("eval/qa_set.jsonl")
DATASET_NAME = "agentic-rag-eval"


def main() -> None:
    client = Client()
    records = [json.loads(line) for line in QA.open(encoding="utf-8")]

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists; reusing it.")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="QA pairs for evaluating the agentic RAG assistant.",
        )
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": r["question"]} for r in records],
            outputs=[{"reference": r["reference_answer"]} for r in records],
        )
        print(f"Created dataset '{DATASET_NAME}' with {len(records)} examples.")

    print("View it in LangSmith → Datasets & Experiments.")


if __name__ == "__main__":
    main()
