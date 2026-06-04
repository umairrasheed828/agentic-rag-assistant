import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.agent.llm import get_llm  # reuse your configured LLM (also loads .env)

EVAL_RUN = Path("eval/eval_run.jsonl")


def main() -> None:
    records = [json.loads(line) for line in EVAL_RUN.open(encoding="utf-8")]

    # Map your fields to RAGAS's expected names
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )

    evaluator_llm = LangchainLLMWrapper(get_llm())
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextRecall(), FactualCorrectness()],
        llm=evaluator_llm,
    )
    print(result)

    result.to_pandas().to_csv("eval/ragas_results.csv", index=False)
    print("\nPer-question scores saved to eval/ragas_results.csv")


if __name__ == "__main__":
    main()
