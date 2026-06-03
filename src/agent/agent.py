from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.llm import get_llm
from src.retrieve.retriever import retrieve
from src.store.pgvector_store import fetch_all_chunks


@tool
def search_papers(query: str) -> str:
    """Search the research-paper knowledge base for passages relevant to the query."""
    chunks = retrieve(query, k=5)
    return "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)


@tool
def corpus_overview() -> str:
    """Describe the knowledge base: how many papers it contains and sample titles.
    Use for meta questions like 'what topics do you cover' or 'how many papers do you have',
    NOT for questions about the papers' actual content."""
    titles = sorted({c["title"] for c in fetch_all_chunks()})
    sample = "\n".join(f"- {t}" for t in titles[:10])
    return f"The knowledge base contains {len(titles)} papers. Sample titles:\n{sample}"


SYSTEM = SystemMessage(
    "You are a research assistant. Use the search_papers tool to find relevant "
    "information before answering questions about AI research. Answer based on what "
    "the tool returns, cite paper titles, and say you don't know if nothing relevant "
    "comes back. For simple greetings or chit-chat, just reply directly."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools = get_llm().bind_tools([search_papers, corpus_overview])


def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([search_papers, corpus_overview]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # agent -> tools OR end
    graph.add_edge("tools", "agent")  # tool result loops back
    return graph.compile()


agent_graph = build_agent()


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How do you evaluate large language models?"
    result = agent_graph.invoke({"messages": [("human", q)]})
    print(result["messages"][-1].content)
