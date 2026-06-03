from src.agent.llm import get_llm
from src.retrieve.retriever import retrieve

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question \
using the context below, which contains excerpts from research papers. Synthesize \
across multiple excerpts to give a useful, complete answer. Cite paper titles when \
relevant. Only say you don't know if the context is clearly unrelated to the question.

Context:
{context}"""


def answer(question: str, k: int = 5) -> str:
    chunks = retrieve(question, k=k)
    context = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)

    llm = get_llm()
    messages = [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
    response = llm.invoke(messages)
    return str(response.content)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How do you evaluate large language models?"
    print(f"Q: {q}\n")
    print(answer(q))
