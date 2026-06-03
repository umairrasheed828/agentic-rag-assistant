from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.llm import get_llm
from src.agent.rag import SYSTEM_PROMPT
from src.retrieve.retriever import retrieve


class RAGState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state: RAGState) -> dict:
    chunks = retrieve(state["question"], k=5)
    context = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    return {"context": context}


def generate_node(state: RAGState) -> dict:
    llm = get_llm()
    messages = [
        ("system", SYSTEM_PROMPT.format(context=state["context"])),
        ("human", state["question"]),
    ]
    response = llm.invoke(messages)
    return {"answer": str(response.content)}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_graph()


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How do you evaluate large language models?"
    result = rag_graph.invoke({"question": q})
    print(result["answer"])
